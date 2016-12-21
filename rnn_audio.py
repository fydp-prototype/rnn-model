import numpy as np
import tensorflow as tf
from datagen import get_training_input, get_training_output


# num_classes = 130  # 64 + 1 frequencies between 0 and 16000/2 stepped by 128, times 2 for real and imag values


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def declare_graph(num_steps, batch_size, state_size=32, learning_rate=0.1, num_inputs_per_step=130):
    reset_graph()

    x = tf.placeholder(tf.float32, [batch_size, num_steps, num_inputs_per_step], name='input_placeholder')  # 20x10
    y = tf.placeholder(tf.float32, [batch_size, num_steps, num_inputs_per_step], name='labels_placeholder')
    init_state = tf.zeros([batch_size, state_size])

    # rnn_inputs is a list of num_steps tensors with shape [batch_size, num_inputs_per_step]
    rnn_inputs = tf.unpack(x, axis=1)

    cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    rnn_outputs, final_state = tf.nn.rnn(cell, rnn_inputs, initial_state=init_state)

    # logits and predictions
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_inputs_per_step])
        b = tf.get_variable('b', [num_inputs_per_step], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    # predictions = [tf.nn.softmax(logit) for logit in logits]
    # predictions = [logit for logit in logits]
    predictions = logits
    # print(tf.shape(predictions))


    # Turn our y placeholder into a list labels
    # y_as_list = [tf.squeeze(i) for i in tf.split(1, num_steps, y)]
    # print(tf.shape(y_as_list))

    # loss_weights = [tf.ones([batch_size]) for _ in range(num_steps)]
    # losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)
    print(tf.shape(y))
    losses = [tf.reduce_mean(tf.square(predictions[i] - y[:, i])) for i in xrange(len(predictions))]

    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x=x,
        y=y,
        init_state=init_state,
        final_state=final_state,
        total_loss=total_loss,
        train_step=train_step,
        preds=predictions,
        saver=tf.train.Saver()
    )


# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_x, raw_y, batch_size, num_steps):
    data_length = len(raw_x)  # 1026
    print(type(raw_x))

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.ndarray)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.ndarray)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
        print(type(data_x))
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        print(type(x))
        print(type(y))
        yield (x, y)


def gen_epochs(num_epochs, num_steps, batch_size):
    for i in range(num_epochs):
        yield gen_batch(get_training_input(), get_training_output(), batch_size, num_steps)


def train_network(g, num_epochs, num_steps, batch_size, verbose=True, save=False):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for epoch_num, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            training_state = None
            for step, (X, Y) in enumerate(epoch):
                if training_state is not None:
                    feed_dict = {g['x']: X, g['y']: np.zeros((20,10,130)), g['init_state']: training_state}
                else:
                    feed_dict = {g['x']: np.zeros((20,10,130)), g['y']: np.zeros((20,10,130))}

                training_loss, training_state, _ = sess.run([g['total_loss'],
                                                             g['final_state'],
                                                             g['train_step']],
                                                            feed_dict)
            if verbose:
                print("Total loss at epoch", epoch_num,
                      ":", training_loss)
            training_losses.append(training_loss)

        if save:
            save_path = g['saver'].save(sess, "/tmp/model-rnn-audio.ckpt")
            print("saved to %s" % save_path)

    return training_losses


def decode_output(ans):
    output = 0
    multiplier = 1
    for i in range(len(ans)):
        output += ans[i] * multiplier
        multiplier *= 2
    return output


def generate_test_output(g):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, "/tmp/model-rnn-audio.ckpt")

        state = None
        test_input = get_training_input()
        ans = []

        for block_index in range(len(test_input)):
            if state is not None:
                feed_dict = {g['x']: [[test_input[block_index]]], g['init_state']: state}
            else:
                feed_dict = {g['x']: [[test_input[block_index]]]}

            preds, state = sess.run([g['preds'], g['final_state']], feed_dict)

            p = np.squeeze(preds)
            ans.append(p)
        return ans


def train():
    graph = declare_graph(num_steps=10, batch_size=20)
    training_losses = train_network(graph, 10, num_steps=10, batch_size=20, save=True)
    import matplotlib.pyplot as plt
    plt.plot(training_losses)
    plt.show()


def test():
    graph = declare_graph(num_steps=1, batch_size=1)

    # for x in range(200, 400):
    #     for y in range(200, 400):
    #         ans = generate_test_output(graph, x, y, 10)
    #         sum = decode_output(ans)
    #         print(str(x) + " + " + str(y) + " = " + str(sum))
    #         print(ans)
    #         assert (sum == x + y)


train()
# test()
