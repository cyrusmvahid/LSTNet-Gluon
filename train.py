from __future__ import print_function

import argparse

import mxnet as mx
from mxnet import nd, gluon, autograd

from dataset import TimeSeriesData
from model import LSTNet
import time
import multiprocessing as mp


def train(file_path, out_path):
    ts_data = TimeSeriesData(file_path, window=24*7, horizon=24)
    ctx = mx.gpu(0)
    min_gpu = 4
    num_gpus = min(min_gpu, mx.context.num_gpus())
    multi_ctx = [mx.gpu(i) for i in range(num_gpus)]
    #multi_ctx = [mx.gpu(0), mx.gpu(1)]

    net = LSTNet(
        num_series=ts_data.num_series,
        conv_hid=100,
        gru_hid=100,
        skip_gru_hid=5,
        skip=24,
        ar_window=24)
    l1 = gluon.loss.L1Loss()

    net.initialize(init=mx.init.Xavier(factor_type="in", magnitude=2.34), ctx=multi_ctx, force_reinit=True)

    trainer = gluon.Trainer(net.collect_params(),
                            optimizer='adam',
                            optimizer_params={'learning_rate': 0.001 * num_gpus, 'clip_gradient': 10.})


    batch_size = 129 * num_gpus
    train_data_loader = gluon.data.DataLoader(
        ts_data.train, batch_size=batch_size, shuffle=True, num_workers=mp.cpu_count(), last_batch='discard')

    #scale = nd.array(ts_data.scale, ctx)
    #scale = ts_data.scale.as_in_context(ctx)
    epochs = 20
  #  loss = None
    print("Training Start")
    for e in range(epochs):
        epoch_loss = mx.nd.zeros((1,), ctx)
        num_iter = 0

        #i = 0
        training_start_time = time.time()
        for i, (data, label) in enumerate(train_data_loader):
            epoch_start_time = time.time()
            #data = data.as_in_context(ctx)
            data = gluon.utils.split_and_load(data=data, ctx_list=multi_ctx)
            #label = label.as_in_context(ctx)
            label = gluon.utils.split_and_load(data=label, ctx_list=multi_ctx)
            losses = []
            outputs = []
#            if loss is not None:
 #               loss.wait_to_read()
            with autograd.record():
                for X, Y in zip(data, label):
                    z = net(X)
                    loss = l1(z, Y)
                    losses.append(loss)
                    outputs.append(z)
            autograd.backward(losses)
            trainer.step(batch_size)
            epoch_loss = epoch_loss + loss.mean()
            num_iter += 1
            #i += 1
            nd.waitall()
        print("Epoch {:3d}; batch {:3d} : epoch loss {:.4}; TIME:{}".format(e, i, epoch_loss.asscalar() / num_iter, time.time()-epoch_start_time))
    print("TRAINING TIME: {}", time.time()-training_start_time)
    net.save_parameters(out_path)
    print("Training End")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTNet Time series forecasting')
    parser.add_argument('--data', type=str, required=True,
                        help='path of the data file')
    parser.add_argument('--out', type=str, required=True,
                        help='path of the trained network output')
    args = parser.parse_args()

    exit(train(args.data, args.out))
