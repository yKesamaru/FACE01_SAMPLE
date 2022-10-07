#cython: language_level=3
"""CHECK SYSTEM INFORMATION
This module is EXPERIMENTAL
"""  
"""TODO: #32 リファクタリングと要件再定義
"""
from GPUtil import getGPUs
from psutil import cpu_count, cpu_freq, virtual_memory

from .api import Dlib_api
from .Calc import Cal
from .logger import Logger

Dlib_api_obj = Dlib_api()
Cal_obj = Cal()


class System_check:

    def __init__(self, log_level: str = 'info') -> None:
        # Setup logger: common way
        self.log_level: str = log_level
        import os.path
        name: str = __name__
        dir: str = os.path.dirname(__file__)
        parent_dir, _ = os.path.split(dir)

        self.logger = Logger(self.log_level).logger(name, parent_dir)
    

        Cal_obj.cal_specify_date(self.logger)


    def system_check(self, CONFIG):
    # lock
        with open("SystemCheckLock", "w") as f:
            f.write('')
        self.logger.info("FACE01の推奨動作環境を満たしているかシステムチェックを実行します")
        self.logger.info("- Python version check")
        major_ver, minor_ver_1, minor_ver_2 = CONFIG["Python_version"].split('.', maxsplit = 3)
        if (version_info < (int(major_ver), int(minor_ver_1), int(minor_ver_2))):
            self.logger.warning("警告: Python 3.8.10以降を使用してください")
            exit(0)
        else:
            self.logger.info(f"  [OK] {str(version)}")
        # CPU
        self.logger.info("- CPU check")
        if cpu_freq().max < float(CONFIG["cpu_freq"]) * 1_000 or cpu_count(logical=False) < int(CONFIG["cpu_count"]):
            self.logger.warning("CPU性能が足りません")
            self.logger.warning("処理速度が実用に達しない恐れがあります")
            self.logger.warning("終了します")
            exit(0)
        else:
            self.logger.info(f"  [OK] {str(cpu_freq().max)[0] + '.' +  str(cpu_freq().max)[1:3]}GHz")
            self.logger.info(f"  [OK] {cpu_count(logical=False)}core")
        # MEMORY
        self.logger.info("- Memory check")
        if virtual_memory().total < int(CONFIG["memory"]) * 1_000_000_000:
            self.logger.warning("メモリーが足りません")
            self.logger.warning("少なくとも4GByte以上が必要です")
            self.logger.warning("終了します")
            exit(0)
        else:
            if int(virtual_memory().total) < 10:
                self.logger.info(f"  [OK] {str(virtual_memory().total)[0]}GByte")
            else:
                self.logger.info(f"  [OK] {str(virtual_memory().total)[0:2]}GByte")
        # GPU
        self.logger.info("- CUDA devices check")
        if CONFIG["gpu_check"] == True:
            if Dlib_api_obj.dlib.cuda.get_num_devices() == 0:
                self.logger.warning("CUDAが有効なデバイスが見つかりません")
                self.logger.warning("終了します")
                exit(0)
            else:
                self.logger.info(f"  [OK] cuda devices: {Dlib_api_obj.dlib.cuda.get_num_devices()}")

            # Dlib build check: CUDA
            self.logger.info("- Dlib build check: CUDA")
            if Dlib_api_obj.dlib.DLIB_USE_CUDA == False:
                self.logger.warning("dlibビルド時にCUDAが有効化されていません")
                self.logger.warning("終了します")
                exit(0)
            else:
                self.logger.info(f"  [OK] DLIB_USE_CUDA: True")

            # Dlib build check: BLAS
            self.logger.info("- Dlib build check: BLAS, LAPACK")
            if Dlib_api_obj.dlib.DLIB_USE_BLAS == False or Dlib_api_obj.dlib.DLIB_USE_LAPACK == False:
                self.logger.warning("BLASまたはLAPACKのいずれか、あるいは両方がインストールされていません")
                self.logger.warning("パッケージマネージャーでインストールしてください")
                self.logger.warning("\tCUBLAS native runtime libraries(Basic Linear Algebra Subroutines: 基本線形代数サブルーチン)")
                self.logger.warning("\tLAPACK バージョン 3.X(線形代数演算を行う総合的な FORTRAN ライブラリ)")
                self.logger.warning("インストール後にdlibを改めて再インストールしてください")
                self.logger.warning("終了します")
                exit(0)
            else:
                self.logger.info("  [OK] DLIB_USE_BLAS, LAPACK: True")

            # VRAM check
            self.logger.info("- VRAM check")
            for gpu in getGPUs():
                gpu_memory = gpu.memoryTotal
                gpu_name = gpu.name
            if gpu_memory < 3000:
                self.logger.warning("GPUデバイスの性能が足りません")
                self.logger.warning(f"現在のGPUデバイス: {gpu_name}")
                self.logger.warning("NVIDIA GeForce GTX 1660 Ti以上をお使いください")
                self.logger.warning("終了します")
                exit(0)
            else:
                if int(gpu_memory) < 9999:
                    self.logger.info(f"  [OK] VRAM: {str(int(gpu_memory))[0]}GByte")
                else:
                    self.logger.info(f"  [OK] VRAM: {str(int(gpu_memory))[0:2]}GByte")
                self.logger.info(f"  [OK] GPU device: {gpu_name}")

        self.logger.info("  ** System check: Done **\n")
