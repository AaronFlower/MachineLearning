#!/usr/bin/env python
#coding:utf-8

# f(x) = x^4 - 3x^3 + 2
# f'(x) = 4x^3 -9x^2

precision = 0.00001
alpha = 0.01
cur_x = 6 # the algorithm starts at x = 6
previous_step_size = 1 + precision

df = lambda x: 4 * x ** 3 - 9 * x ** 2

# 当变化收敛的步长小于精度时，就可以认为是收敛了
while previous_step_size > precision:
    prev_x = cur_x
    cur_x += -alpha * df(prev_x)
    previous_step_size = abs(cur_x - prev_x)

print("The local minimum occurs at %f" % cur_x)

