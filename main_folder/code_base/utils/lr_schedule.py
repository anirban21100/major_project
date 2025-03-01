import tensorflow as tf

def lr_schedule(epoch):
  """
  Returns a custom learning rate that decreases as epochs progress.
  """
  learning_rate = 0.01
  if epoch > 10:
    learning_rate = 0.003
  if epoch > 15:
    learning_rate = 0.002
  if epoch > 25:
    learning_rate = 0.001
  if epoch > 30:
    learning_rate = 0.0001

  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate