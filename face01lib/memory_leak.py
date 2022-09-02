import linecache
import tracemalloc

import psutil
from pympler import muppy, summary, tracker


class Memory_leak:
    __doc__ = "Output the result of the tracemalloc and Pympler module with formatted."
    __author__ = "Yoshitsugu Kesamaru (yKesamaru)"
    __email__ = "y.kesamaru@tokai-kaoninsho.com"
    __maintainer__ = "yKesamaru"
    __license__ = "MIT"
    __copyright__ = "yKesamaru"
    __version__ = "v0.0.1"

    def __init__(self, limit: int, key_type: str, **kwargs: int) -> None:
        """Output the result of the tracemalloc module with formatted.

        Args:
            limit:(int) Limit output lines.
            key_type:(str) Select 'lineno' or 'traceback' output. Defaults to 'lineno'.
            nframe:(int, optional) This can be specified only when key_type is 'traceback'. Defaults to 5.
        
        For reference, See bellow
        - tracemalloc
            - [En]
                - https://docs.python.org/3/library/tracemalloc.html#
            - [ja]
                - https://docs.python.org/ja/3/library/tracemalloc.html#
        - Pympler
            - https://pympler.readthedocs.io/en/latest/
        """    
        self.limit: int = limit
        self.key_type: str = key_type
        self.nframe: int | None = kwargs.get('nframe')

        self.used_memory1 = psutil.virtual_memory().used

        print("-" * 30)
        print(f"Called 'memory_leak.py' with '{self.key_type}' mode...")
        print("-" * 30)


    def __display_line(self, snapshot, key_type='lineno', limit=5) -> None:
        self.snapshot = snapshot
        self.key_type = key_type
        self.limit = limit

        snapshot = self.snapshot.filter_traces(
            (
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
                tracemalloc.Filter(False, tracemalloc.__file__),
                tracemalloc.Filter(False, "<unknown>"),
            )
        )
        stats = snapshot.statistics(self.key_type)

        print("\nTop %s lines" % self.limit)

        for index, stat in enumerate(stats[:self.limit], 1):
            frame = stat.traceback[0]
            print("#%s: File:%s\n    Line: %s\n    Size: %.1f MiB"
                % (index, frame.filename, frame.lineno, stat.size / 1024 / 1048))
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print('    %s' % line)
            linecache.clearcache()
            print("-" * 5)

        other = stats[self.limit:]
        if other:
            size: int = sum(stat.size for stat in other)
            print("%s, Other: %.1f MiB" % (len(other), size / 1024 / 1048))
        total: int = sum(stat.size for stat in stats)
        print("Total allocated size: %.1f MiB" % (total / 1024 / 1048))


    def __display_traceback(self, stats, key_type='traceback', limit=5) -> None:
        self.stats = stats
        self.key_type = key_type
        self.limit = limit

        print("\nTop %s traceback" % self.limit)

        index = 1
        for stat in self.stats[:self.limit]:
            print(f"#{index}\n{stat}")
            for line in stat.traceback.format():  # traceback.format(limit=5, most_recent_first=True)
                if line:
                    print(f"    {line}")
            print("-" * 5)
            index += 1

        other = stats[self.limit:]
        if other:
            size = sum(stat.size for stat in other)
            print("%s, Other: %.1f MiB" % (len(other), size / 1024 / 1048))
        total = sum(stat.size for stat in stats)
        print("Total allocated size: %.1f MiB" % (total / 1024 / 1048))


    def memory_leak_analyze_start(self) -> None:
        if self.key_type == 'traceback':
            if isinstance(self.nframe, int):
                tracemalloc.start(self.nframe)
                self.snapshot1 = tracemalloc.take_snapshot().filter_traces(
                    (
                        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                        tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
                        tracemalloc.Filter(False, tracemalloc.__file__),
                        tracemalloc.Filter(False, "<unknown>"),
                    )
                )
            else:
                print("nframe: int")
                exit(0)
        
        elif self.key_type == 'lineno':
            tracemalloc.start()
        
        else:
            print("The argument 'key_type' must specify either 'lineno' or 'traceback'."); exit()

        self.tr = tracker.SummaryTracker()


    def memory_leak_analyze_stop(self) -> None:
        if self.key_type == 'traceback':
            snapshot2 = tracemalloc.take_snapshot()
            stats = snapshot2.compare_to(self.snapshot1, 'traceback')
            self.__display_traceback(stats, 'traceback', self.limit)
        
        elif self.key_type == 'lineno':
            snapshot = tracemalloc.take_snapshot()
            self.__display_line(snapshot, 'lineno', self.limit)

        tracemalloc_memory = tracemalloc.get_tracemalloc_memory()
        used_memory2 = psutil.virtual_memory().used
        used_memory = used_memory2 - self.used_memory1 - tracemalloc_memory
        print("-" * 30)
        print("Used Memory: %.1f GiB" % (used_memory / 1024 /1048 / 1074))
        print("-" * 30)
        # pympler
        print("\nPympler report")
        print("<Summary>")
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        summary.print_(sum1)
        print("-" * 30)
        print("<Summary_diff>")
        self.tr.print_diff()
