#cython: language_level=3

from GPUtil import getGPUs
from psutil import cpu_count, cpu_freq, virtual_memory
from .logger import Logger
from os.path import dirname, exists
from .Calc import Cal

Cal_obj = Cal()


name: str = __name__
dir: str = dirname(__file__)
logger = Logger().logger(name, dir, 'info')


class System_check:
    """CHECK SYSTEM INFORMATION"
    """    
    def __init__(self) -> None:
        Cal_obj.cal_specify_date(logger)

    @staticmethod
    def system_check(args_dict):
    # lock
        with open("SystemCheckLock", "w") as f:
            f.write('')
        logger.info("FACE01の推奨動作環境を満たしているかシステムチェックを実行します")
        logger.info("- Python version check")
        major_ver, minor_ver_1, minor_ver_2 = args_dict["Python_version"].split('.', maxsplit = 3)
        if (version_info < (int(major_ver), int(minor_ver_1), int(minor_ver_2))):
            logger.warning("警告: Python 3.8.10以降を使用してください")
            exit(0)
        else:
            logger.info(f"  [OK] {str(version)}")
        # CPU
        logger.info("- CPU check")
        if cpu_freq().max < float(args_dict["cpu_freq"]) * 1_000 or cpu_count(logical=False) < int(args_dict["cpu_count"]):
            logger.warning("CPU性能が足りません")
            logger.warning("処理速度が実用に達しない恐れがあります")
            logger.warning("終了します")
            exit(0)
        else:
            logger.info(f"  [OK] {str(cpu_freq().max)[0] + '.' +  str(cpu_freq().max)[1:3]}GHz")
            logger.info(f"  [OK] {cpu_count(logical=False)}core")
        # MEMORY
        logger.info("- Memory check")
        if virtual_memory().total < int(args_dict["memory"]) * 1_000_000_000:
            logger.warning("メモリーが足りません")
            logger.warning("少なくとも4GByte以上が必要です")
            logger.warning("終了します")
            exit(0)
        else:
            if int(virtual_memory().total) < 10:
                logger.info(f"  [OK] {str(virtual_memory().total)[0]}GByte")
            else:
                logger.info(f"  [OK] {str(virtual_memory().total)[0:2]}GByte")
        # GPU
        logger.info("- CUDA devices check")
        if args_dict["gpu_check"] == True:
            if Dlib_api_obj.dlib.cuda.get_num_devices() == 0:
                logger.warning("CUDAが有効なデバイスが見つかりません")
                logger.warning("終了します")
                exit(0)
            else:
                logger.info(f"  [OK] cuda devices: {Dlib_api_obj.dlib.cuda.get_num_devices()}")

            # Dlib build check: CUDA
            logger.info("- Dlib build check: CUDA")
            if Dlib_api_obj.dlib.DLIB_USE_CUDA == False:
                logger.warning("dlibビルド時にCUDAが有効化されていません")
                logger.warning("終了します")
                exit(0)
            else:
                logger.info(f"  [OK] DLIB_USE_CUDA: True")

            # Dlib build check: BLAS
            logger.info("- Dlib build check: BLAS, LAPACK")
            if Dlib_api_obj.dlib.DLIB_USE_BLAS == False or Dlib_api_obj.dlib.DLIB_USE_LAPACK == False:
                logger.warning("BLASまたはLAPACKのいずれか、あるいは両方がインストールされていません")
                logger.warning("パッケージマネージャーでインストールしてください")
                logger.warning("\tCUBLAS native runtime libraries(Basic Linear Algebra Subroutines: 基本線形代数サブルーチン)")
                logger.warning("\tLAPACK バージョン 3.X(線形代数演算を行う総合的な FORTRAN ライブラリ)")
                logger.warning("インストール後にdlibを改めて再インストールしてください")
                logger.warning("終了します")
                exit(0)
            else:
                logger.info("  [OK] DLIB_USE_BLAS, LAPACK: True")

            # VRAM check
            logger.info("- VRAM check")
            for gpu in getGPUs():
                gpu_memory = gpu.memoryTotal
                gpu_name = gpu.name
            if gpu_memory < 3000:
                logger.warning("GPUデバイスの性能が足りません")
                logger.warning(f"現在のGPUデバイス: {gpu_name}")
                logger.warning("NVIDIA GeForce GTX 1660 Ti以上をお使いください")
                logger.warning("終了します")
                exit(0)
            else:
                if int(gpu_memory) < 9999:
                    logger.info(f"  [OK] VRAM: {str(int(gpu_memory))[0]}GByte")
                else:
                    logger.info(f"  [OK] VRAM: {str(int(gpu_memory))[0:2]}GByte")
                logger.info(f"  [OK] GPU device: {gpu_name}")

        logger.info("  ** System check: Done **\n")