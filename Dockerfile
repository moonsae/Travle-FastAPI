# 기존 이미지
FROM python:3.9-slim

# Java 설치 및 Python 빌드 필수 패키지 설치
RUN apt-get update && \
    apt-get install -y default-jdk python3-distutils python3-setuptools python3-pip python3-wheel build-essential && \
    apt-get clean

# JAVA_HOME 환경 변수 설정
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# 앱 디렉토리 설정
WORKDIR /app

# 필요한 파일 복사
COPY . /app

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 앱 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
