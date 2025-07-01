import os

# 데이터 경로 설정
images_path = '../../dataset/train/detection/imagesTr'
annotations_path ='../../dataset/train/detection/labelsTr'

# 디렉토리 존재 여부 확인
if not os.path.exists(images_path) or not os.path.exists(annotations_path):
    print("Error: 이미지 경로 또는 어노테이션 경로가 올바르지 않습니다.")
else:
    # 학습 및 평가 파일 실행
    os.system('python train.py')
