"""DEBUG: #23 Memory leak
See bellow
https://qiita.com/hnw/items/3e01f60eb190f748539a
https://docs.python.org/ja/3/library/tracemalloc.html
"""


import linecache
import tracemalloc
import psutil

class Memory_leak:
    def display_top(self, snapshot, key_type='lineno', limit=10):
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

        print("\n")
        print("⭐️" * 3)
        used_memory = psutil.virtual_memory().used
        print("USED MEMORY TOTAL: %.1f GiB" % (used_memory / 1024 /1048 / 1074))
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
    
    def memory_leak_analyze_start(self):
        tracemalloc.start()

    def memory_leak_analyze_stop(self):
        snapshot = tracemalloc.take_snapshot()
        self.display_top(snapshot)



# tracemalloc.start(20)
# snapshot1 = tracemalloc.take_snapshot()

# snapshot2 = tracemalloc.take_snapshot()
# top_stats = snapshot2.compare_to(snapshot1, 'traceback')
# # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
# # snapshot = tracemalloc.take_snapshot()
# # top_stats = snapshot.statistics('traceback')
# num = 1
# for stat in top_stats[:10]:
#     print(num); num += 1
#     print(f'stat: {stat}')
#     for line in stat.traceback.format():
#         print(f'line: {line}')
#     print("=" * 20, "\n")