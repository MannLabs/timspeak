class Logger:

    def _get_logs_directory(self) -> str:
        import os
        logs_directory = os.path.join(os.getcwd(), 'logs_timspeak')
        if not os.path.exists(logs_directory):
            os.makedirs(logs_directory)
        return logs_directory

    def _get_logs_file_name(self) -> str:
        import os
        import time
        time_stamp = time.localtime()
        current_date = '-'.join([
            f'{time_stamp.tm_year:04}',
            f'{time_stamp.tm_mon:02}',
            f'{time_stamp.tm_mday:02}'
        ])
        current_time = '_'.join([
            f'{time_stamp.tm_hour:02}',
            f'{time_stamp.tm_min:02}',
            f'{time_stamp.tm_sec:02}'
        ])
        logs_directory = self._get_logs_directory()
        log_file_name = os.path.join(logs_directory, f'execution_{current_date}__{current_time}.log')
        return log_file_name

    def _get_formatter(self) -> str:
        import logging
        return logging.Formatter('%(asctime)s> %(message)s', "%Y-%m-%d %H:%M:%S")

    def _get_platform_info(self, logger) -> None:
        import alphatims
        import timspeak.platform_utilities.platform_settings
        platform = timspeak.platform_utilities.platform_settings.Platform()
        logger.info('---------- PLATFORM INFO----------')
        logger.info(f'platform:        {platform.platform}')
        logger.info(f'system:          {platform.system}')
        logger.info(f'release:         {platform.release}')
        logger.info(f'version:         {platform.version}')
        logger.info(f'machine:         {platform.machine}')
        logger.info(f'processor:       {platform.processor}')
        logger.info(f'cpu_count:       {platform.cpu_count}')
        logger.info(f'ram_available:   {platform.ram_available}')
        logger.info(f'ram_total:       {platform.ram_total}')
        logger.info(f'separator:       {platform.separator}')
        logger.info(f'python:          {platform.python}')
        logger.info(f'alphatims:       {alphatims.__version__}')
        logger.info(f'timspeak:      {platform.timspeak}')

    def __init__(
        self,
        log_level: int = 20,
        stream: bool = True,
    ) -> None:
        import sys
        import logging
        root_logger = logging.getLogger()
        print(type(root_logger))
        root_logger.setLevel(log_level)
        while root_logger.hasHandlers():
            root_logger.removeHandler(root_logger.handlers[0])
        formatter = self._get_formatter()
        if stream:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(log_level)
            stream_handler.setFormatter(formatter)
            root_logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(self._get_logs_file_name(), mode="w")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f'---------- INITIALIZE LOGGER ----------')
        self._get_platform_info(root_logger)
        self.root_logger = root_logger
