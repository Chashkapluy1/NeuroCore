import asyncio
import logging
import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import aiofiles
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Настройка структурированного логирования
# Создаем форматтер, который включает контекст (время, имя логгера, уровень)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Обработчик INFO-сообщений для стандартных операций (вывод в консоль)
info_handler = logging.StreamHandler()
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(log_formatter)

# Отдельный обработчик ERROR-сообщений для фиксации сбоев в файл
error_handler = logging.FileHandler('downloader_errors.log', encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(log_formatter)

logger = logging.getLogger('AsyncDownloader')
logger.setLevel(logging.DEBUG)  # Захватываем все, фильтрация происходит на уровне обработчиков
logger.addHandler(info_handler)
logger.addHandler(error_handler)


@dataclass
class DownloadResult:
    """
    Дата-класс для хранения результатов скачивания.
    Позволяет унифицировать контракт возвращаемых данных между разными реализациями.

    Attributes:
        url (str): Исходный URL целевого файла.
        success (bool): Флаг успешного завершения операции.
        file_path (Optional[Path]): Путь к сохраненному файлу (если успешно).
        error (Optional[str]): Описание ошибки (если произошел сбой).
    """
    url: str
    success: bool
    file_path: Optional[Path] = None
    error: Optional[str] = None


class BaseDownloader(ABC):
    """
    Абстрактный базовый класс (ABC) загрузчика.
    Определяет единый интерфейс (контракт) для всех последующих реализаций.
    """
    def __init__(self, concurrency_limit: int, output_directory: Path | str, mock: bool = False):
        """
        Инициализация базового загрузчика.

        Args:
            concurrency_limit (int): Максимальное количество одновременных сетевых соединений (лимит конкурентности).
            output_directory (Path | str): Директория для сохранения скачанных файлов.
            mock (bool): Флаг включения mock-режима для симуляции сети без реальных HTTP-запросов.
        """
        self.concurrency_limit = concurrency_limit
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.mock = mock

    @abstractmethod
    def download_one(self, url: str) -> DownloadResult:
        """
        Синхронный фасад для скачивания одного файла.

        Args:
            url (str): URL для скачивания.

        Returns:
            DownloadResult: Результат операции.
        """
        pass

    @abstractmethod
    def download_all(self, urls: List[str]) -> List[DownloadResult]:
        """
        Синхронный фасад для массового скачивания списка файлов.

        Args:
            urls (List[str]): Список URL для обработки.

        Returns:
            List[DownloadResult]: Список результатов для каждого переданного URL.
        """
        pass


class AsyncDownloader(BaseDownloader):
    """
    Асинхронная реализация BaseDownloader на базе aiohttp и asyncio.TaskGroup.
    Обеспечивает высокую производительность I/O операций с минимальным потреблением ресурсов ОС.
    Включает систему плавного завершения (Graceful Shutdown) и умные повторные попытки (Exponential Backoff).
    """
    
    def __init__(self, concurrency_limit: int, output_directory: Path | str, 
                 progress_callback: Optional[Callable[[str], None]] = None,
                 mock: bool = False):
        """
        Инициализация AsyncDownloader.

        Args:
            concurrency_limit (int): Лимит конкурентности корутин.
            output_directory (Path | str): Целевая директория.
            progress_callback (Optional[Callable[[str], None]]): Optional коллбэк для отслеживания прогресса.
            mock (bool): Использовать ли mock-режим.
        """
        super().__init__(concurrency_limit, output_directory, mock)
        self.progress_callback = progress_callback
        # Event для сигнализации всем корутинам о необходимости прервать выполнение
        self._shutdown_event = asyncio.Event()

    # Декоратор Tenacity: Повторять до 5 раз.
    # Стратегия ожидания: Экспоненциальная (1с, 2с, 4с, 8с), максимум 10 секунд между попытками.
    # Повторять только при ошибках соединения и таймаутах, фатальные ошибки (напр. 404) выбрасываются сразу.
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError, asyncio.TimeoutError)),
        reraise=True
    )
    async def _perform_request(self, url: str, session: aiohttp.ClientSession, file_path: Path):
        """
        Изолированная корутина для выполнения самого сетевого запроса.
        Обернута в логику повторных попыток (retry) для повышения надежности (Reliability).
        """
        if self.mock:
            # Симуляция сетевой задержки (Network Latency) без реальных HTTP запросов
            await asyncio.sleep(1)
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(b"MOCK_IMAGE_DATA")
            return

        # Таймаут на всю операцию чтения (30 секунд)
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            response.raise_for_status()
            
            # Используем aiofiles для неблокирующего (non-blocking) дискового I/O
            async with aiofiles.open(file_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    # Проверяем сигнал прерывания во время скачивания больших файлов
                    if self._shutdown_event.is_set():
                        raise asyncio.CancelledError("Запрошено завершение работы (Shutdown) во время скачивания")
                    await f.write(chunk)

    async def _download_file(self, url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> DownloadResult:
        """
        Внутренняя корутина для скачивания отдельного файла с учетом лимитов конкурентности через Семафор.
        """
        async with semaphore:
            # Если получен сигнал завершения до начала скачивания - возвращаем ошибку
            if self._shutdown_event.is_set():
                return DownloadResult(url=url, success=False, error="Запрошено завершение работы (Shutdown)")

            # Извлекаем базовое имя файла или генерируем хэш, если URL не содержит явного имени
            original_filename = url.split('/')[-1]
            if not original_filename or '?' in original_filename:
                original_filename = f"downloaded_{hash(url)}"
                
            file_path = self.output_directory / original_filename
            
            try:
                # Делегируем выполнение функции, обернутой в Tenacity
                await self._perform_request(url, session, file_path)
                            
                if self.progress_callback:
                    self.progress_callback(url)
                else:
                    logger.info(f"Успешно скачано{' (MOCK)' if self.mock else ''}: {url}")
                    
                return DownloadResult(url=url, success=True, file_path=file_path)
                
            except asyncio.CancelledError:
                logger.warning(f"Скачивание отменено (Cancelled) для: {url}")
                return DownloadResult(url=url, success=False, error="Отменено пользователем")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Сбой скачивания {url} после всех попыток: {error_msg}")
                return DownloadResult(url=url, success=False, error=error_msg)

    async def _async_download_all(self, urls: List[str]) -> List[DownloadResult]:
        """
        Главный оркестратор корутин. Управляет сессией, пулом соединений и абстракцией TaskGroup.
        """
        # TCPConnector ограничивает пул соединений на транспортном уровне
        connector = aiohttp.TCPConnector(limit=self.concurrency_limit)
        
        # Семафор ограничивает количество активных задач (корутин) в памяти интерпретатора
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        
        results: List[DownloadResult] = []
        tasks = []
        
        # Настройка обработчиков сигналов ОС (SIGINT/Ctrl+C) для Graceful Shutdown
        loop = asyncio.get_running_loop()
        
        def handle_shutdown():
            logger.warning("Получен сигнал прерывания (SIGINT). Отмена ожидающих задач...")
            self._shutdown_event.set()
            for t in tasks:
                if not t.done():
                    t.cancel()

        try:
            # Кроссплатформенная поддержка обработчиков (полноценно работает в Unix, имеет фоллбэк для Windows)
            loop.add_signal_handler(signal.SIGINT, handle_shutdown)
            loop.add_signal_handler(signal.SIGTERM, handle_shutdown)
        except NotImplementedError:
            # Fallback для Windows
            signal.signal(signal.SIGINT, lambda sig, frame: handle_shutdown())

        async with aiohttp.ClientSession(connector=connector) as session:
            try:
                # В Python 3.11+ используется TaskGroup вместо устаревшего asyncio.gather
                async with asyncio.TaskGroup() as tg:
                    for url in urls:
                        if self._shutdown_event.is_set():
                            break
                        task = tg.create_task(self._download_file(url, session, semaphore))
                        tasks.append(task)
                        
                results = [task.result() for task in tasks if task.done()]
                
            except* Exception as eg:
                # ExceptionGroup перехватывает фатальные исключения внутри TaskGroup
                logger.error(f"TaskGroup столкнулся с критическими исключениями (ExceptionGroup): {eg.exceptions}")
                results = [task.result() for task in tasks if task.done() and not task.exception()]
            except asyncio.CancelledError:
                logger.warning("Главный событийный цикл был отменен.")
                results = [task.result() for task in tasks if task.done()]
                
        return results

    def download_one(self, url: str) -> DownloadResult:
        """
        Синхронный фасад для запуска единичной загрузки.
        Скрывает asyncio.run от вызывающего кода.
        """
        try:
            return asyncio.run(self._async_download_all([url]))[0]
        except KeyboardInterrupt:
            logger.warning("Процесс прерван пользователем (KeyboardInterrupt).")
            return DownloadResult(url=url, success=False, error="KeyboardInterrupt")

    def download_all(self, urls: List[str]) -> List[DownloadResult]:
        """
        Синхронный фасад для массовой загрузки.
        """
        try:
            return asyncio.run(self._async_download_all(urls))
        except KeyboardInterrupt:
            logger.warning("Процесс прерван пользователем. Возвращаем частичные результаты.")
            return []


if __name__ == "__main__":
    downloader = AsyncDownloader(concurrency_limit=3, output_directory="./test_downloads", mock=True)
    test_urls = [f"https://mock.url/{i}.png" for i in range(10)]
    print("Запуск mock-scrapping. Нажмите Ctrl+C для тестирования Graceful Shutdown.")
    res = downloader.download_all(test_urls)
    print(f"Завершено: успешно {len([r for r in res if r.success])} из {len(test_urls)}")
