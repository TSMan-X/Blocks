from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='lltm_cpp',						# 编译后的链接库名称
    ext_modules=[
        cpp_extension.CppExtension(
            'lltm_cpp', ['csrc/lltm.cpp']		       # 待编译文件，及编译函数
        )
    ],
    cmdclass={						       # 执行编译命令设置
        'build_ext': cpp_extension.BuildExtension.with_options(no_python_abi_suffix=True)
    }
)
