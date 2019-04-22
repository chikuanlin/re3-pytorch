# python3 training/training_v2.py -m 10000 -n 2 -u 0 -b 64 -l 1e-5 >> log
# python3 training/training_v2.py -m 20000 -n 2 -u 0 -b 64 -l 1e-6 >> log
# python3 training/training_v2.py -m 10000 -n 2 -u 0 -b 64 -l 1e-6 >> log
# python3 training/training_v2.py -m 5000 -n 4 -u 0 -b 32 -l 1e-6 >> log
# python3 training/training_v2.py -m 10000 -n 4 -u 0 -b 32 -l 1e-6 >> log
# python3 training/training_v2.py -m 10000 -n 4 -u 0.1 -b 32 -l 1e-6 >> log
# python3 training/training_v2.py -m 15000 -n 4 -u 0.1 -b 32 -l 1e-6 >> log
# python3 training/training_v2.py -m 20000 -n 8 -u 0.1 -b 16 -l 1e-6 >> log
# python3 training/training_v2.py -m 20000 -n 8 -u 0.2 -b 16 -l 1e-6 >> log
# python3 training/training_v2.py -m 20000 -n 16 -u 0.2 -b 8 -l 5e-7 >> log
# python3 training/training_v2.py -m 20000 -n 16 -u 0.25 -b 8 -l 5e-7 >> log

python3 training/training_v2.py -m 5000 -n 32 -u 0.25 -b 4 -l 5e-7 >> log
git add .
git commit -m '-m 5000 -n 16 -u 0.25 -b 8 -l 5e-7'
git push https://pikelcw:pikelcw110@github.com/chikuanlin/re3-pytorch.git

python3 training/training_v2.py -m 5000 -n 32 -u 0.35 -b 4 -l 5e-7 >> log
git add .
git commit -m '-m 5000 -n 16 -u 0.35 -b 8 -l 5e-7'
git push https://pikelcw:pikelcw110@github.com/chikuanlin/re3-pytorch.git

python3 training/training_v2.py -m 5000 -n 32 -u 0.5 -b 4 -l 5e-7 >> log
git add .
git commit -m '-m 5000 -n 16 -u 0.5 -b 8 -l 5e-7'
git push https://pikelcw:pikelcw110@github.com/chikuanlin/re3-pytorch.git

python3 training/training_v2.py -m 5000 -n 32 -u 0.65 -b 4 -l 5e-7 >> log
git add .
git commit -m '-m 5000 -n 16 -u 0.65 -b 8 -l 5e-7'
git push https://pikelcw:pikelcw110@github.com/chikuanlin/re3-pytorch.git

python3 training/training_v2.py -m 10000 -n 32 -u 0.8 -b 4 -l 5e-7 >> log
git add .
git commit -m '-m 10000 -n 16 -u 0.8 -b 8 -l 5e-7'
git push https://pikelcw:pikelcw110@github.com/chikuanlin/re3-pytorch.git

shutdown -h now


