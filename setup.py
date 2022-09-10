from setuptools import setup


def get_requirements(requirements_path: str = "requirements.txt"):
    with open(requirements_path) as fp:
        return [
            x.strip() for x in fp.read().split("\n") if not x.startswith("#")
        ]

def get_long_description():
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
    return long_description
        
setup(name='KorSEC', # 패키지 명
    version='1.0.2',
    author='seoyeon9695',
    author_email='seoyeon9695@gmail.com',
    description='Korean space error correction package',
    long_description = get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/seoyeon9646/KorSEC',
    license='MIT', # MIT에서 정한 표준 라이센스 따른다
    python_requires='>=3',
    install_requires=get_requirements(), # 패키지 사용을 위해 필요한 추가 설치 패키지
    include_package_data=True,
    packages=['KorSEC'] # 패키지가 들어있는 폴더들
)