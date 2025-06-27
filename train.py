import os
from datetime import datetime
os.chdir(r"E:\AMFformer-main")

from best import *

data_train, afn_tr, hdr_tr = get_acdc(acdc_data_train)
X_tr, Y_tr, A_tr, E_tr = data_train[0], data_train[1], data_train[2], data_train[3]

data_val, afn_val, hdr_val = get_acdc(acdc_data_validation)
X_val, Y_val, A_val, E_val = data_val[0], data_val[1], data_val[2], data_val[3]

data_test, afn_ts, hdr_ts = get_acdc(acdc_data_test)
X_ts, Y_ts, A_ts, E_ts = data_test[0], data_test[1], data_test[2], data_test[3]

Y_tr_cat = convert_masks(Y_tr)
Y_val_cat = convert_masks(Y_val)
E_tr_cat = convert_masks(E_tr)
E_val_cat = convert_masks(E_val)

bs = 2

gen_tr = unite_gen(X_tr, Y_tr_cat[:, ::4, ::4, :], Y_tr_cat[:, ::2, ::2, :], Y_tr_cat, E_tr_cat, bs, "training")
gen_val = unite_gen(X_val, Y_val_cat[:, ::4, ::4, :], Y_val_cat[:, ::2, ::2, :], Y_val_cat, E_val_cat, bs, "validation")

net = AMFformer(X_tr)

log_path = r"/root/tf-logs/train//" + datetime.now().strftime("%Y%m%d-%H%M%S")

cb_tb = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)

warm_epochs = 8
stage1_epochs = 50
stage2_epochs = 120

cb_rlr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', mode='min', patience=5, factor=0.5, min_delta=0.001, verbose=1
)

save_path = r"E:\AMFformer-main\re.h5"
cb_ckpt = keras.callbacks.ModelCheckpoint(
    save_path, monitor="val_loss", save_best_only=True, save_weights_only=True
)

cb_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", restore_best_weights=True, min_delta=0.001, patience=5
)

loss_fn = "binary_crossentropy"
init_lr = 1e-3
optimizer = tf.keras.optimizers.Nadam(learning_rate=init_lr)

warm_batches = warm_epochs * len(X_tr) // bs
cb_warmup = WarmUpLearningRateScheduler(warm_batches, init_lr=init_lr)

net.compile(
    optimizer=optimizer,
    loss=[loss_fn, loss_fn, loss_fn],
    loss_weights=[0.14*0.8, 0.29*0.8, 0.57*0.8, 0.2],
)

net.fit(
    gen_tr,
    steps_per_epoch=len(X_tr) // bs,
    epochs=stage1_epochs,
    callbacks=[cb_warmup],
)

net.fit(
    gen_tr,
    validation_data=gen_val,
    steps_per_epoch=len(X_tr) // bs,
    validation_steps=len(X_val) // bs,
    epochs=stage2_epochs,
    callbacks=[cb_rlr, cb_ckpt, cb_tb],
)

net.save_weights(save_path)

with open("E:\AMFformer-main\output.txt", "w") as f:
    preds = net.predict(X_ts, batch_size=1)
    uniq = np.unique(np.argmax(preds[2], axis=3))
    print(uniq, file=f)
    print(uniq)
    score = np.round(np.array(metrics(Y_ts[:, :, :, -1], np.argmax(preds[2], axis=3), 0)), 4)
    print(score, file=f)
    print(score)
    avg = np.round(score.mean(), 4)
    print(f"The average is: {avg}", file=f)
    print(f"The average is: {avg}")
