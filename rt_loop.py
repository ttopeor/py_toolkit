from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor as APSchedulerThreadPoolExecutor

import threading
import time


class RTLoop:
    def __init__(self, freq=50, print_frequency=False):
        """
        Initialize the RTLoop class.

        :param freq: Loop frequency in Hz, default is 50
        :param print_frequency: If True, prints loop frequency and elapsed time in loop_spin, default is False
        """
        self.freq = freq
        self.count = 0
        self.hz = 0  # Current loop frequency
        self.time_from_start = 0  # Elapsed time in milliseconds
        self.scheduler = BackgroundScheduler(
            executors={
                'default': APSchedulerThreadPoolExecutor(max_workers=1)
            },
            job_defaults={
                'coalesce': True,
                'max_instances': 1,
                'misfire_grace_time': 1
            }
        )
        self._lock = threading.Lock()  # Thread lock to ensure thread safety
        self._start_time = None  # To track the start time
        self._last_update_time = None  # To track the last frequency update time
        self.print_frequency = print_frequency  # Control printing in loop_spin
        self._loop_job_id = 'loop_job'

    def setup(self):
        """
        One-time initialization setup, add scheduled tasks, and start the scheduler.
        Subclasses can override this method to add custom initialization content.
        """
        # Record the start time using perf_counter for higher precision
        self._start_time = time.perf_counter()
        self._last_update_time = self._start_time

        self.scheduler.add_job(self.loop, 'interval',
                               seconds=1/self.freq,
                               id=self._loop_job_id,
                               max_instances=1,
                               coalesce=True)
        self.scheduler.add_job(self.update_frequency, 'interval',
                               seconds=1, id='update_freq_job',
                               max_instances=1, coalesce=True)
        self.scheduler.add_job(self.update_time_from_start, 'interval',
                               seconds=0.1, id='time_job',
                               max_instances=1, coalesce=True)

        # Start the scheduler
        self.scheduler.start()
        print("Scheduler started.")

    def loop(self):
        """
        Define the task to be executed periodically.
        Subclasses need to override this method to implement custom loop tasks.
        """
        with self._lock:
            self.count += 1
        # Add other operations that need to be executed in the loop here
        # For example, sensor reading, data processing, etc.

    def update_frequency(self):
        """
        Calculate and update the current loop frequency.
        """
        current_time = time.perf_counter()
        with self._lock:
            duration = current_time - self._last_update_time
            if duration > 0:
                self.hz = self.count / duration
            else:
                self.hz = 0
            # Debugging information
            # print(f"Debug: Count={self.count}, Duration={duration:.4f} seconds, Calculated Hz={self.hz:.2f}")
            self.count = 0
            self._last_update_time = current_time

    def update_time_from_start(self):
        """
        Update the elapsed time since the start of the loop in milliseconds.
        """
        if self._start_time is not None:
            elapsed_time_sec = time.perf_counter() - self._start_time
            with self._lock:
                # Convert to milliseconds
                self.time_from_start = int(elapsed_time_sec * 1000)

    def get_hz(self):
        """
        Get the current loop frequency in Hz.

        :return: Current loop frequency as a float
        """
        with self._lock:
            return self.hz

    def get_time_from_start(self):
        """
        Get the elapsed time since the start of the loop in milliseconds.

        :return: Elapsed time in milliseconds as an integer
        """
        with self._lock:
            return self.time_from_start

    def pause_loop(self):
        try:
            self.scheduler.pause_job(self._loop_job_id)
        except Exception:
            pass

    def resume_loop(self):
        try:
            self.scheduler.resume_job(self._loop_job_id)
        except Exception:
            pass

    def shutdown(self):
        """
        Shutdown the scheduler to ensure the program exits cleanly.
        """
        self.scheduler.shutdown()
        print("Scheduler shutdown.")

    def loop_spin(self):
        """
        Keep the main thread running and handle shutdown on interrupt.
        Additionally, allows the main thread to perform tasks like accessing hz.
        If print_frequency is True, prints the current Hz and elapsed time.
        """
        try:
            while True:
                time.sleep(1)  # Adjust sleep duration as needed
                if self.print_frequency:
                    current_hz = self.get_hz()
                    elapsed_time = self.get_time_from_start()
                    print(
                        f"Accessed from main thread - Current Hz: {current_hz:.2f}, Elapsed Time: {elapsed_time} ms")
        except (KeyboardInterrupt, SystemExit):
            self.shutdown()
