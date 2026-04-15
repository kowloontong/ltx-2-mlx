uv run ltx-2-mlx generate \
  --prompt "Extreme close-up of a cyber-wuxia swordsman..." \
  --hq \
  --stage1-steps 20 \
  --stage2-steps 5 \
  --cfg-scale 3.5 \
  --height 576 \
  --width 832 \
  --model dgrauet/ltx-2.3-mlx-q8 \
  -o hq.mp4
