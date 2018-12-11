#!/usr/bin/env python3

import tensorflow as tf

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print("#" * 70)
print()
print("It works!")
print()
print("Check to see if the container is using your GPU's (it should have")
print(" a device:GPU entry in the debug output).")
print()
print("#" * 70)
print()
