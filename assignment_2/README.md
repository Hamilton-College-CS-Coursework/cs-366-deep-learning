Assignment 2 Read.me

When in the remote server, 

1) cd into the repo folder titled "assignment_2"
2) cd into "src" folder located in "assignment_2"
3) run on the cmd line:

   python train.py
   
5) best_chkpt.ckpt should now be located in the "checkpoints" folder in "assignment_2"
6) run on the cmd line: enter all the arguments like so...

if you want to see 12 image segmentation samples...

python main.py \
  --root ~/hamilton/cpsci366/data \
  --split test \
  --classes trimap \
  --resize 512 \
  --n 12 \
  --seed 42 \
  --save-dir samples \
  --ckpt checkpoints/best_chkpt.ckpt \
  --alpha 0.45


if you want to see one image segmentation sample...

python main.py \
  --ckpt checkpoints/unetpp_best.ckpt \
  --predict-image ./test_images/pet01.png \
  --classes binary \
  --resize 512 \
  --alpha 0.5
