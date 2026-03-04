import concurrent.futures
import logging
import signal
import threading
import time
from typing import Callable, List, Optional
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from async_downloader import BaseDownloader, DownloadResult

# Идентичная конфигурация логирования как в AsyncDownloader
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

info_handler = logging.StreamHandler()
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(log_formatter)

error_handler = logging.FileHandler('downloader_errors.log', encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(log_formatter)

logger = logging.getLogger('ThreadedDownloader')
logger.setLevel(logging.DEBUG)
logger.addHandler(info_handler)
logger.addHandler(error_handler)


class ThreadedDownloader(BaseDownloader):
    """
    Синхронная реализация BaseDownloader с использованием пула потоков (concurrent.futures.ThreadPoolExecutor)
    и библиотеки requests. Включает механизм Graceful Shutdown (threading.Event) и умные повторные
    запросы (Tenacity) для управления сетью.
    """
    def __init__(self, concurrency_limit: int, output_directory: Path | str, 
                 progress_callback: Optional[Callable[[str], None]] = None,
                 mock: bool = False):
        """
        Инициализация ThreadedDownloader.

        Args:
            concurrency_limit (int): Максимальное количество активных потоков пула.
            output_directory (Path | str): Целевая директория сохранения.
            progress_callback (Optional[Callable[[str], None]]): Коллбэк для индикации прогресса.
            mock (bool): Запуск без реальных сетевых запросов.
        """
        super().__init__(concurrency_limit, output_directory, mock)
        self.progress_callback = progress_callback
        
        # Потокобезопасный (thread-safe) счетчик успешных операций
        self._success_count = 0
        # Мьютекс (Lock) требуется при мутации разделяемого состояния из нескольких потоков
        self._lock = threading.Lock()
        
        # Сигнальное событие (Event) для оповещения рабочих потоков (worker threads) об остановке
        self._shutdown_event = threading.Event()
        
        # requests.Session() значительно ускоряет работу за счет пула соединений (connection pool) 
        # и поддержки HTTP Keep-Alive, избавляя от накладных расходов на TCP-хендшейки.
        self.session = requests.Session()
        
        # Адаптер жестко лимитирует TCP-соединения на уровне urllib3 до значения concurrency_limit
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=concurrency_limit,
            pool_maxsize=concurrency_limit
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def _increment_success(self):
        """Потокобезопасная инкрементация счетчика через блокировку (Lock)."""
        with self._lock:
            self._success_count += 1

    # Декоратор Tenacity: Экспоненциальный откат (Exponential Backoff). 
    # Работает только при ошибках сокетов/таймаутах HTTP-слоя.
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True
    )
    def _perform_request(self, url: str, file_path: Path):
        """Изолированная функция сетевого запроса, обернутая для безопасных повторов."""
        if self.mock:
            # Симуляция сетевой задержки блокирует поток, отдавая ресурсы ОС (Context Switch)
            time.sleep(1)
            with open(file_path, 'wb') as f:
                f.write(b"MOCK_IMAGE_DATA")
            return

        with self.session.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    # Периодическая проверка флага Shutdown обеспечивает быстрый обрыв загрузки тяжелых файлов
                    if self._shutdown_event.is_set():
                        raise InterruptedError("Загрузка прервана (Shutdown) при чтении чанков")
                    if chunk:
                        f.write(chunk)

    def _download_file(self, url: str) -> DownloadResult:
        """Рабочая функция (Worker Function) для выполнения непосредственно пулом потоков."""
        
        # Сразу обрываем выполнение задания, если сигнал Shutdown поступил до старта потока
        if self._shutdown_event.is_set():
            return DownloadResult(url=url, success=False, error="Остановлено (Shutdown) до начала")
            
        original_filename = url.split('/')[-1]
        if not original_filename or '?' in original_filename:
            original_filename = f"downloaded_{hash(url)}"
            
        file_path = self.output_directory / original_filename
        
        try:
            # Вызов функции сетевого I/O, освобождающей Global Interpreter Lock (GIL)
            self._perform_request(url, file_path)
            
            self._increment_success()
            
            if self.progress_callback:
                self.progress_callback(url)
            else:
                logger.info(f"Успешно скачано{' (MOCK)' if self.mock else ''}: {url}")
                
            return DownloadResult(url=url, success=True, file_path=file_path)
            
        except InterruptedError:
            logger.warning(f"Скачивание отменено (Cancelled) для: {url}")
            return DownloadResult(url=url, success=False, error="Отменено пользователем")
        except Exception as e:
            error_msg = str(e)
            # При разборе Error Handler'ом эта запись попадет напрямую в файл логов (downloader_errors.log)
            logger.error(f"Сбой скачивания {url} после попыток: {error_msg}")
            return DownloadResult(url=url, success=False, error=error_msg)

    def download_one(self, url: str) -> DownloadResult:
        """Скачивание одного файла синхронно (без оркестрации пула)."""
        try:
            return self._download_file(url)
        except KeyboardInterrupt:
            self._shutdown_event.set()
            return DownloadResult(url=url, success=False, error="KeyboardInterrupt")

    def download_all(self, urls: List[str]) -> List[DownloadResult]:
        """Оркестрация пула потоков (ThreadPoolExecutor) для загрузки списка."""
        results = []
        
        # Сброс метрик
        with self._lock:
            self._success_count = 0
            
        # Защита обработчиков ОС
        original_sigint = signal.getsignal(signal.SIGINT)
        
        def handle_shutdown(signum, frame):
            logger.warning("Активирован SIGINT (Graceful Shutdown)! Прерываем потоки...")
            self._shutdown_event.set()
            # Сброс обработчика позволяет жестко убить процесс при повторном нажатии Ctrl+C
            signal.signal(signal.SIGINT, original_sigint)

        signal.signal(signal.SIGINT, handle_shutdown)

        try:
            # Использование ThreadPoolExecutor гарантирует автоматическое управление потоками
            # и чистое высвобождение ресурсов при выходе из контекста `with`.
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency_limit) as executor:
                # submits() используем вместо map(), чтобы получить объекты Future для возможной отмены
                future_to_url = {executor.submit(self._download_file, url): url for url in urls}
                
                # Итерируемся по задачам по мере их асинхронного завершения
                for future in concurrent.futures.as_completed(future_to_url):
                    if self._shutdown_event.is_set():
                        # Активная очистка очереди Executor'а (Future Cancellation), если задан Shutdown
                        for pending in future_to_url:
                            if not pending.done():
                                pending.cancel()
                        break
                    
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Критическая ошибка Future: {e}")

        except Exception as main_e:
            logger.error(f"Сбой главного пула (ThreadPool): {main_e}")
        finally:
            signal.signal(signal.SIGINT, original_sigint)

        return results


if __name__ == "__main__":
    downloader = ThreadedDownloader(concurrency_limit=3, output_directory="./test_sync", mock=True)
    test_urls = [f"https://mock.url/{i}.png" for i in range(10)]
    print("Запуск mock-scrapping потоками. Нажмите Ctrl+C для тестирования Graceful Shutdown.")
    res = downloader.download_all(test_urls)
    print(f"Завершено: успешно {len([r for r in res if r.success])} из {len(test_urls)}")
