# 001 이하의 wav 목록 만들기
find ./001 -type f -name "*.wav" > wav.list

# 800개씩 잘라서 batch_aa, batch_ab ... 파일 생성
split -l 800 wav.list batch_

# 각 배치를 순차 업로드 (메타데이터는 최초에 한 번 같이 올리거나 마지막에)
for f in batch_*; do
  xargs -a "$f" -d '\n' git add --          # 해당 배치 wav만 stage
  git commit -m "Add wav batch $(basename "$f")"
  git push origin main
  sleep 90                                  # 잠깐 쉬어 HF rate limit 피하기 (필요시 늘리세요)
done
