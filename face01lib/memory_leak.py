"""DEBUG: #23 Memory leak
For reference, See bellow
https://qiita.com/hnw/items/3e01f60eb190f748539a
https://docs.python.org/ja/3/library/tracemalloc.html
"""


import linecache
import sys
import tracemalloc

import psutil


class Memory_leak:
    def __init__(self) -> None:
        global used_memory1
        used_memory1 = psutil.virtual_memory().used
        print("=" * 20)
        print("⭐️" * 3, "Memory usage before start: %.1f GiB" % (used_memory1 / 1024 /1048 / 1074))

    def display_line(self, snapshot, key_type='lineno', limit=5):
        self.snapshot = snapshot
        self.key_type = key_type
        self.limit = limit

        snapshot = self.snapshot.filter_traces(
            (
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<unknown>"),
            )
        )
        top_stats = snapshot.statistics(self.key_type)

        print("Top %s lines" % self.limit)

        for index, stat in enumerate(top_stats[:self.limit], 1):
            frame = stat.traceback[0]
            print("#%s: File:%s\n    Line: %s\n    Size: %.1f MiB"
                % (index, frame.filename, frame.lineno, stat.size / 1024 / 1048))
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print('    %s' % line)
            print("-" * 5)

        other = top_stats[self.limit:]
        if other:
            size = sum(stat.size for stat in other)
            print("%s箇所, その他: %.1f MiB" % (len(other), size / 1024 / 1048))
        total = sum(stat.size for stat in top_stats)
        print("Total allocated size: %.1f MiB" % (total / 1024 / 1048))
        print("=" * 20)
        print("\n")
    
    def display_traceback(self, stats):
        self.stats = stats
        num = 1
        for stat in self.stats[:5]:
            print("\n", "-" * 20)
            print("# ", num); num += 1
            print(f'stat: {stat}')
            for line in stat.traceback.format():
                print(f'      {line}')
        print("=" * 20)
        print("\n")

    def memory_leak_analyze_start(self, line_or_traceback: str = 'line'):
        self.line_or_traceback = line_or_traceback
        if self.line_or_traceback == 'traceback':
            tracemalloc.start(20)
            self.snapshot1 = tracemalloc.take_snapshot()
            pass
        elif self.line_or_traceback == 'line':
            tracemalloc.start()
        else:
            print("The argument 'line_or_traceback' must specify either 'line' or 'traceback'."); exit()

    def memory_leak_analyze_stop(self, line_or_traceback: str = 'line'):
        self.line_or_traceback = line_or_traceback
        if self.line_or_traceback == 'traceback':
            snapshot2 = tracemalloc.take_snapshot()
            stats = snapshot2.compare_to(self.snapshot1, 'traceback')
            self.display_traceback(stats)
        elif self.line_or_traceback == 'line':
            snapshot = tracemalloc.take_snapshot()
            self.display_line(snapshot)

        used_memory2 = psutil.virtual_memory().used
        used_memory = used_memory2 - used_memory1
        print("⭐️" * 3, "Used Memory: %.1f GiB" % (used_memory / 1024 /1048 / 1074))