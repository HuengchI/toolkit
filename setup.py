import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()
setuptools.setup(
    name="toolkit",  # 模块名称
    version="1.0",  # 当前版本
    author="huengchi",  # 作者
    # author_email="wupeiqi@live.com",  # 作者邮箱
    # description="一个非常NB的包",  # 模块简介
    # long_description=long_description,  # 模块详细介绍
    # long_description_content_type="text/markdown",  # 模块详细介绍格式
    url="https://github.com/HuengchI/toolkit",  # 模块github地址
    packages=setuptools.find_packages(),  # 自动找到项目中导入的模块
    # 模块相关的元数据
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    # 依赖模块
    install_requires=[
        # 'pillow',
    ],
    python_requires='>=3',
)