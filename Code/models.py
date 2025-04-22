"""Models module"""

import os
import psutil
import tensorflow as tf
from _losses_FEDERICO import real_mse_loss, fake_mse_loss, cycle_loss, id_loss
from _metrics_FEDERICO import MSE_Metric
from _callbacks_FEDERICO import __CheckpointManager__


def conv_block(signal_len,
               input_filters,
               output_filters,
               kernel_size,
               strides=1,
               apply_instancenorm=False,
               name="ConvBlock"):

    input_layer = tf.keras.layers.Input(shape=[signal_len, input_filters],
                                        name=f"{name}_Input")

    result = tf.keras.layers.Conv1D(filters=output_filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding='same',
                                    use_bias=False,
                                    name=f"{name}_conv{kernel_size}")(input_layer)

    if apply_instancenorm:
        result = tf.keras.layers.GroupNormalization(groups=-1,
                                                    name=f"{name}_InstNorm")(result)

    result = tf.keras.layers.Activation(tf.nn.swish,
                                        name=f"{name}_swish"
                                        )(result)

    return tf.keras.Model(inputs=input_layer,  
                          outputs=result,
                          name=name)


def inception_block(signal_len,
                    input_filters,
                    output_filters,
                    receptive_fields,
                    name="InceptionLayer"):
    
    """Concatenate multiple conv_block with different stride"""

    input_layer = tf.keras.layers.Input(shape=[signal_len, input_filters],
                                        name=f"{name}_Input")

    n = len(receptive_fields)
    filter_per_branch = []
    if output_filters % n == 0:
        for i in range(n):
            filter_per_branch.append(output_filters // n)
    else:
        zp = output_filters % n
        pp = output_filters // n
        for i in range(n):
            if i >= zp:
                filter_per_branch.append(pp)
            else:
                filter_per_branch.append(pp + 1)

    conv_block_list = []

    for kernel_size, num_filter in zip(receptive_fields, filter_per_branch):
        conv_block_list.append(conv_block(signal_len=signal_len,
                                          input_filters=input_filters,
                                          output_filters=num_filter,
                                          kernel_size=kernel_size,
                                          strides=1,
                                          apply_instancenorm=True,
                                          name=f"{name}_ConvBlock{kernel_size}")(input_layer))

    result = tf.keras.layers.concatenate(conv_block_list,
                                         name=f"{name}_Concatenate")

    return tf.keras.Model(inputs=input_layer,
                          outputs=result,
                          name=name)


def BRDAE_block(signal_len,
                n_channels=3,
                recurrent_units=100,
                ):
    input_layer = tf.keras.layers.Input(shape=[signal_len, n_channels],
                                        name="BRDAE_Input")
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        recurrent_units,
        return_sequences=True,
        dropout=0.5,
    ),
        name="BRDAE_Bidirectional")(input_layer)
    x = tf.keras.layers.Dense(n_channels,
                              activation="linear",
                              name="BRDAE_Output")(x)

    return tf.keras.Model(inputs=input_layer,
                          outputs=x,
                          name="BRDAE")


def Inception_Unet(signal_len,
                   n_channels,
                   n_layers,
                   filters,
                   inception_layers=-1,  
                   inception_receptive_fields=None,  
                   traditional_block_kernel=3,  
                   recurrent_skip=False,  
                   recurrent_exit=False,  
                   exit_recurrent_units=None,  
                   output_channels=None,
                   variable_kernel_size=False): 

    if filters % 2 != 0:
        raise ValueError("Filters must be a power of 2.")

    if inception_layers == -1:
        inception_layers = n_layers
    elif inception_layers > n_layers:
        raise ValueError(
            f"inception_layers ({inception_layers}) cannot be greater than n_layers ({n_layers}).")

    if inception_layers > 0 and inception_receptive_fields is None:
        raise ValueError(
            "inception_receptive_filed must be specified in a list when inception_layers is not zero.")

    if recurrent_exit is True and exit_recurrent_units is None:
        raise ValueError(
            "exit_recurrent_unit must be set as an integer value when recurrent_exit is True.")

    if output_channels is None:
        output_channels = n_channels

    traditional_layers = n_layers - inception_layers
    if variable_kernel_size:
        if (traditional_layers-inception_layers-1)*4 >= traditional_block_kernel:
            raise ValueError(
                "When Variable kernel size is set to True, (traditional_layers-inception_layers)*4 must be less than traditional_block_kernel")

    input_layer = tf.keras.layers.Input(shape=[signal_len, n_channels],
                                        name="Input_layer")

    x = input_layer
    skips = []

    # Encoder Path
    for i in range(n_layers):
        if variable_kernel_size:
            cutter = (i-1)*4
        else:
            cutter = 0
        if i == 0:
            input_filters = n_channels
            output_filters = filters
        else:
            input_filters = filters * (2**(i - 1))
            output_filters = input_filters * 2

        current_signal_len = signal_len // (2**i)

        if i < inception_layers:
            x = inception_block(signal_len=current_signal_len,
                                input_filters=input_filters,
                                output_filters=output_filters,
                                receptive_fields=inception_receptive_fields,
                                name=f"InceptionDown{i+1}")(x)
        else:
           #print(f'layer{i+1} ha kenrel: {traditional_block_kernel - cutter}')
            x = conv_block(signal_len=current_signal_len,
                           input_filters=input_filters,
                           output_filters=output_filters,
                           kernel_size=traditional_block_kernel - cutter,
                           strides=1,
                           apply_instancenorm=True,
                           name=f"TraditionalDown{i+1}")(x)
        skips.append(x)
        if recurrent_skip:
            skips[-1] = tf.keras.layers.GRU(units=output_filters,
                                            # activation="linear", # default is tanh
                                            return_sequences=True,
                                            name=f"SkipConnectionGRU{i+1}",
                                            )(skips[-1])

        # x = tf.keras.layers.MaxPool1D(name=f"Downsample{i+1}")(x)
        x = tf.keras.layers.Conv1D(filters=output_filters,
                                   kernel_size=1,
                                   strides=2,
                                   padding='same',
                                   use_bias=False,
                                   name=f"Downsample{i+1}"
                                   )(x)

    # Bottleneck
    bottleneck_filters = filters * (2**i)
    current_signal_len = signal_len//(2**(i+1))
    if traditional_layers == 0:
        x = inception_block(signal_len=current_signal_len,
                            input_filters=bottleneck_filters,
                            output_filters=bottleneck_filters,
                            receptive_fields=inception_receptive_fields,
                            name="InceptionBottleneck")(x)
    else:
        #print(f'layer bottleneck ha kenrel: {traditional_block_kernel - cutter}')
        x = conv_block(signal_len=current_signal_len,
                       input_filters=bottleneck_filters,
                       output_filters=bottleneck_filters,
                       kernel_size=traditional_block_kernel-cutter,
                       strides=1,
                       apply_instancenorm=True,
                       name="Bottleneck")(x)

    # Decoder Path
    for i in reversed(range(n_layers)):
        if variable_kernel_size:
            cutter = (i-1)*4
        else:
            cutter = 0
        if i == 0:
            input_filters = filters * (2**i)
            output_filters = input_filters
        else:
            input_filters = filters * (2**i)
            output_filters = input_filters // 2

        current_signal_len = signal_len // (2**i)

        x = tf.keras.layers.Conv1DTranspose(filters=input_filters,
                                            kernel_size=1,
                                            strides=2,
                                            padding='same',
                                            use_bias=False,
                                            name=f"Upsample{i+1}"
                                            )(x)

        # x = tf.keras.layers.UpSampling1D(name=f"Upsample{i+1}")(x)

        x = tf.keras.layers.Concatenate(name=f"SkipConnection{i+1}")([x,
                                                                      skips[i]])
        x = tf.keras.layers.Conv1D(filters * (2**i),
                                   kernel_size=1,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   name=f"SkipConnectionBottleneck{i+1}")(x)

        if i < inception_layers:
            x = inception_block(signal_len=current_signal_len,
                                input_filters=input_filters,
                                output_filters=output_filters,
                                receptive_fields=inception_receptive_fields,
                                name=f"InceptionUp{i+1}")(x)

        else:
            #print(f'layer{i+1} ha kenrel: {traditional_block_kernel - cutter}')
            x = conv_block(signal_len=current_signal_len,
                           input_filters=input_filters,
                           output_filters=output_filters,
                           kernel_size=traditional_block_kernel - cutter,
                           strides=1,
                           apply_instancenorm=True,
                           name=f"TraditionalUp{i+1}")(x)

    # Last convolution
    x = tf.keras.layers.Conv1D(filters=output_channels,
                               kernel_size=1,
                               strides=1,
                               activation='linear',
                               padding="same",
                               use_bias=True,
                               name="LastConv")(x)

    if recurrent_exit:
        x = BRDAE_block(signal_len=signal_len,
                        n_channels=n_channels,
                        recurrent_units=exit_recurrent_units)(x)

    # Setup Network name
    if inception_receptive_fields is not None and inception_layers != 0:
        inc_layers_str = f"_{inception_layers}inc"

        inc_kernels_str = "-".join([str(x)
                                   for x in inception_receptive_fields])
        inc_kernels_str = f"_{inc_kernels_str}"
    else:
        inc_layers_str = ""
        inc_kernels_str = ""

    if inception_layers < n_layers:
        trad_block_kernel_str = f"_{traditional_block_kernel}k"
    else:
        trad_block_kernel_str = ""

    if recurrent_skip:
        recurrent_skip_str = "_S"
    else:
        recurrent_skip_str = ""

    if recurrent_exit:
        recurrent_exit_str = f"_E{exit_recurrent_units}"
    else:
        recurrent_exit_str = ""
    if variable_kernel_size:
        var_k_size = "_Var"
    else:
        var_k_size = ""

    network_name = f"{signal_len}x{n_channels}_{n_layers}l_{filters}f{inc_layers_str}{inc_kernels_str}{trad_block_kernel_str}{recurrent_skip_str}{recurrent_exit_str}{var_k_size}"

    return tf.keras.Model(inputs=input_layer,
                          outputs=x,
                          name=network_name)


def Discriminator(signal_len,
                  n_channels,
                  n_layers,
                  filters,
                  kernel_size,
                  variable_kernel_size=False,
                  ):
    # initializer = tf.random_normal_initializer(0., 0.02, seed=42)

    input_layer = tf.keras.layers.Input(shape=[signal_len, n_channels],
                                        name="Input_layer")

    x = input_layer

    if variable_kernel_size:
        if (n_layers-1)*2 >= kernel_size:
            raise ValueError(
                "When Variable kernel size is set to True, n_layers*2 must be less than disc_kernel_size")

    for i in range(n_layers):
        current_signal_len = signal_len // (2**i)

        if variable_kernel_size:
            cutter = i*2
        else:
            cutter = 0

        if i == 0:
            input_filters = n_channels
            output_filters = filters
            #print(f'layer{i+1} ha kenrel: {kernel_size - cutter}')

            x = conv_block(signal_len=current_signal_len,
                           input_filters=input_filters,
                           output_filters=output_filters,
                           kernel_size=kernel_size - cutter,
                           strides=2,
                           apply_instancenorm=True,
                           name=f"ConvBlock{i+1}")(x)
        elif i == (n_layers - 1):
            input_filters = filters * (2**(i-1))
            output_filters = input_filters // 2
            #print(f'layer{i+1} ha kenrel: {kernel_size - cutter}')

            x = conv_block(signal_len=current_signal_len,  # substituted signal_len with current_signal_len
                           input_filters=input_filters,
                           output_filters=output_filters,
                           kernel_size=kernel_size-cutter,
                           strides=1,
                           apply_instancenorm=True,
                           name=f"ConvBlock{i+1}")(x)
        else:
            input_filters = filters * (2**(i - 1))
            output_filters = input_filters * 2
            #print(f'layer{i+1} ha kenrel: {kernel_size - cutter}')

            x = conv_block(signal_len=current_signal_len,  # substituted signal_len with current_signal_len
                           input_filters=input_filters,
                           output_filters=output_filters,
                           kernel_size=kernel_size-cutter,
                           strides=2,
                           apply_instancenorm=True,
                           name=f"ConvBlock{i+1}")(x)

    # Last layer
    #print(f'layer{i+1} ha kenrel: {kernel_size - cutter}')

    x = tf.keras.layers.Conv1D(filters=1,
                               kernel_size=kernel_size-cutter,
                               strides=1,
                               padding="same",
                               use_bias=True,
                               activation="sigmoid",
                               name="Output_layer")(x)

    # Setup Network name
    if variable_kernel_size:
        network_name = f"{signal_len}x{n_channels}_{n_layers}l_{filters}f_{kernel_size}k_Var"
    else:
        network_name = f"{signal_len}x{n_channels}_{n_layers}l_{filters}f_{kernel_size}k"

    return tf.keras.Model(inputs=input_layer,
                          outputs=x,
                          name=network_name)


class cycleGAN(tf.keras.Model):
    def __init__(
        self,
        generator_A,
        generator_B,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=1.0,
        lambda_identity=0.5,
        lambda_adv=1.0,
        lambda_MSE=1.0,
        restore_filepath=None
    ):
        super().__init__()
        self.gen_A = generator_A
        self.gen_B = generator_B
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_adv = lambda_adv
        self.lambda_MSE = lambda_MSE
        self.restore_filepath = restore_filepath

        # self.g_params = list(self.gen_A.trainable_variables) + list(self.gen_B.trainable_variables)

        if self.restore_filepath is not None:
            self.compile()

        self.id = f"G{generator_A.name}__D{discriminator_X.name}"

    def compile(
        self,
        gen_A_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                 beta_1=0.5),  # by default beta_1 = 0.9
        gen_B_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                 beta_1=0.5),
        disc_X_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                  beta_1=0.5),
        disc_Y_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                  beta_1=0.5),
    ):
        super().compile()

        # Set Optimizers
        self.gen_A_optimizer = gen_A_optimizer
        # Ensure the optimizer is built with all trainable variables
        self.gen_A_optimizer.build(self.gen_A.trainable_variables)

        self.gen_B_optimizer = gen_B_optimizer
        # Ensure the optimizer is built with all trainable variables
        self.gen_B_optimizer.build(self.gen_B.trainable_variables)

        self.disc_X_optimizer = disc_X_optimizer
        # Ensure the optimizer is built with all trainable variables
        self.disc_X_optimizer.build(self.disc_X.trainable_variables)

        self.disc_Y_optimizer = disc_Y_optimizer
        # Ensure the optimizer is built with all trainable variables
        self.disc_Y_optimizer.build(self.disc_Y.trainable_variables)

        # Set Metrics
        self.gen_A_loss_tracker = tf.keras.metrics.Mean(name="loss_A")
        self.gen_B_loss_tracker = tf.keras.metrics.Mean(name="loss_B")
        self.disc_X_loss_tracker = tf.keras.metrics.Mean(name="loss_X")
        self.disc_Y_loss_tracker = tf.keras.metrics.Mean(name="loss_Y")
        self.MSE = MSE_Metric(name="MSE")

        # Evaluate if restore model
        if self.restore_filepath is not None:
            ckpt_manager = __CheckpointManager__(filepath=self.restore_filepath,
                                                 model=self
                                                 )
            ckpt_manager.restore()

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
            ):
        """Wrapper for tf.keras.Model.fit() method. It creates a customized
        ProgbarLogger object and then pass everything to the original method.
        Without the customized ProgbarLogger, values showed as outputs are
        wrong, because this callback average values between batches by default,
        but we have already computed the correct mean in train_step."""

        if not isinstance(callbacks, tf.keras.callbacks.CallbackList):
            custom_progbar = tf.keras.callbacks.ProgbarLogger(count_mode='steps',
                                                              stateful_metrics=["loss_A",
                                                                                "loss_B",
                                                                                "loss_X",
                                                                                "loss_Y"
                                                                                ])
            if callbacks is None:
                callbacks = [custom_progbar]
            else:
                callbacks.insert(0, custom_progbar)

        log = super().fit(x=x,
                          y=y,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          callbacks=callbacks,
                          validation_split=validation_split,
                          validation_data=validation_data,
                          shuffle=shuffle,
                          class_weight=class_weight,
                          sample_weight=sample_weight,
                          initial_epoch=initial_epoch,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=validation_steps,
                          validation_batch_size=validation_batch_size,
                          validation_freq=validation_freq,
                          max_queue_size=max_queue_size,
                          workers=workers,
                          use_multiprocessing=use_multiprocessing)

        return log

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None,
                 callbacks=None,
                 max_queue_size=10,
                 workers=1,
                 use_multiprocessing=False,
                 return_dict=True,
                 **kwargs
                 ):
        """Wrapper for tf.keras.Model.evaluate() method. It creates a customized
    ProgbarLogger object and then pass everything to the original method.
    Without the customized ProgbarLogger, values showed as outputs are wrong,
    because this callback average values between batches by default, but we
    have already computed the correct mean in test_step."""

        if not isinstance(callbacks, tf.keras.callbacks.CallbackList):
            custom_progbar = tf.keras.callbacks.ProgbarLogger(count_mode='steps',
                                                              stateful_metrics=["loss_A",
                                                                                "loss_B",
                                                                                "loss_X",
                                                                                "loss_Y"
                                                                                ])
            if callbacks is None:
                callbacks = [custom_progbar]
            else:
                callbacks.insert(0, custom_progbar)

        log = super().evaluate(x=x,
                               y=y,
                               batch_size=batch_size,
                               verbose=verbose,
                               sample_weight=sample_weight,
                               steps=steps,
                               callbacks=callbacks,
                               max_queue_size=max_queue_size,
                               workers=workers,
                               use_multiprocessing=use_multiprocessing,
                               return_dict=return_dict,
                               **kwargs)
        return log

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                show_report=False,
                export_report=None,
                report_labels=None,
                ):
        prediction = super().predict(x=x,
                                     batch_size=batch_size,
                                     verbose=verbose,
                                     steps=steps,
                                     callbacks=callbacks,
                                     max_queue_size=max_queue_size,
                                     workers=workers,
                                     use_multiprocessing=use_multiprocessing,
                                     )

        if show_report:
            self.show_report_handler(prediction=prediction,
                                     export_report=export_report,
                                     report_labels=report_labels)

        return prediction

    @ tf.function
    def train_step(self, data):
        real_x, real_y = data
        process = psutil.Process()

        memory = process.memory_info()
        print(memory)

        with tf.GradientTape(persistent=True) as tape:
            # x -> fake_y
            fake_y = self.gen_A(real_x, training=True)
            # y -> fake_x
            fake_x = self.gen_B(real_y, training=True)

            # Cycle x -> fake_y -> cycled_x
            cycled_x = self.gen_B(fake_y, training=True)
            # Cycle y -> fake_x -> cycled_y
            cycled_y = self.gen_A(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_B(real_x, training=True)
            same_y = self.gen_A(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adversarial loss
            adv_loss_A = real_mse_loss(disc_fake_y, self.lambda_adv)
            adv_loss_B = real_mse_loss(disc_fake_x, self.lambda_adv)
            # Generator cycle loss
            cycle_loss_A = cycle_loss(real_y, cycled_y, self.lambda_cycle)
            cycle_loss_B = cycle_loss(real_x, cycled_x, self.lambda_cycle)

            # Generator identity loss
            id_loss_A = id_loss(real_y, same_y, self.lambda_identity)
            id_loss_B = id_loss(real_x, same_x, self.lambda_identity)

            '''mse_loss_A = (tf.reduce_mean(tf.square(real_y - fake_y)) *
                          self.lambda_MSE
                          )

            mse_loss_B = (tf.reduce_mean(tf.square(real_x - fake_x)) *
                          self.lambda_MSE
                          )'''

            # Total generator loss
            gen_A_loss = adv_loss_A + cycle_loss_A + \
                id_loss_A  # + adv_loss_B + cycle_loss_B + id_loss_B
            gen_B_loss = adv_loss_B + cycle_loss_B + \
                id_loss_B  # + adv_loss_A + cycle_loss_A + id_loss_A

            # Discriminator loss
            disc_x_real_loss = real_mse_loss(disc_real_x, self.lambda_adv)
            disc_x_fake_loss = fake_mse_loss(disc_fake_x, self.lambda_adv)
            disc_y_real_loss = real_mse_loss(disc_real_y, self.lambda_adv)
            disc_y_fake_loss = fake_mse_loss(disc_fake_y, self.lambda_adv)
            disc_X_loss = disc_x_fake_loss + disc_x_real_loss
            disc_Y_loss = disc_y_fake_loss + disc_y_real_loss

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss,
                                     self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss,
                                     self.disc_Y.trainable_variables)

        # Get the gradients for the generators
        grads_A = tape.gradient(gen_A_loss, self.gen_A.trainable_variables)
        grads_B = tape.gradient(gen_B_loss, self.gen_B.trainable_variables)

        # Update the weights of the generators
        self.gen_A_optimizer.apply_gradients(zip(grads_A,
                                                 self.gen_A.trainable_variables))
        self.gen_B_optimizer.apply_gradients(zip(grads_B,
                                                 self.gen_B.trainable_variables))

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(zip(disc_X_grads,
                                                  self.disc_X.trainable_variables))
        self.disc_Y_optimizer.apply_gradients(zip(disc_Y_grads,
                                                  self.disc_Y.trainable_variables))

        # Update metric trackers
        self.gen_A_loss_tracker.update_state(gen_A_loss)
        self.gen_B_loss_tracker.update_state(gen_B_loss)
        self.disc_X_loss_tracker.update_state(disc_X_loss)
        self.disc_Y_loss_tracker.update_state(disc_Y_loss)
        self.MSE.update_state(real_y, fake_y)

        return {
            "loss_A": self.gen_A_loss_tracker.result(),
            "loss_B": self.gen_B_loss_tracker.result(),
            "loss_X": self.disc_X_loss_tracker.result(),
            "loss_Y": self.disc_Y_loss_tracker.result(),
            "MSE": self.MSE.result(),
        }

    @ tf.function
    def test_step(self, data):
        real_x, real_y = data

        # x -> fake_y
        fake_y = self.gen_A(real_x, training=False)
        # y -> fake_x
        fake_x = self.gen_B(real_y, training=False)

        # Cycle x -> fake_y -> cycled_x
        cycled_x = self.gen_B(fake_y, training=False)
        # Cycle y -> fake_x -> cycled_y
        cycled_y = self.gen_A(fake_x, training=False)

        # Identity mapping
        same_x = self.gen_B(real_x, training=False)
        same_y = self.gen_A(real_y, training=False)

        # Discriminator output
        disc_real_x = self.disc_X(real_x, training=False)
        disc_fake_x = self.disc_X(fake_x, training=False)

        disc_real_y = self.disc_Y(real_y, training=False)
        disc_fake_y = self.disc_Y(fake_y, training=False)

        # Generator adversarial loss
        adv_loss_A = real_mse_loss(disc_fake_y, self.lambda_adv)
        adv_loss_B = real_mse_loss(disc_fake_x, self.lambda_adv)
        # Generator cycle loss
        cycle_loss_A = cycle_loss(real_y, cycled_y, self.lambda_cycle)
        cycle_loss_B = cycle_loss(real_x, cycled_x, self.lambda_cycle)

        # Generator identity loss
        id_loss_A = id_loss(real_y, same_y, self.lambda_identity)
        id_loss_B = id_loss(real_x, same_x, self.lambda_identity)

        '''mse_loss_A = (tf.reduce_mean(tf.square(real_y - fake_y)) *
                        self.lambda_MSE
                        )

        mse_loss_B = (tf.reduce_mean(tf.square(real_x - fake_x)) *
                        self.lambda_MSE
                        )'''

        # Total generator loss
        gen_A_loss = adv_loss_A + cycle_loss_A + \
            id_loss_A  # + adv_loss_B + cycle_loss_B + id_loss_B
        gen_B_loss = adv_loss_B + cycle_loss_B + \
            id_loss_B  # + adv_loss_A + cycle_loss_A + id_loss_A

        # Discriminator loss
        disc_x_real_loss = real_mse_loss(disc_real_x, self.lambda_adv)
        disc_x_fake_loss = fake_mse_loss(disc_fake_x, self.lambda_adv)
        disc_y_real_loss = real_mse_loss(disc_real_y, self.lambda_adv)
        disc_y_fake_loss = fake_mse_loss(disc_fake_y, self.lambda_adv)
        disc_X_loss = disc_x_fake_loss + disc_x_real_loss
        disc_Y_loss = disc_y_fake_loss + disc_y_real_loss
        # Update metric trackers
        self.gen_A_loss_tracker.update_state(gen_A_loss)
        self.gen_B_loss_tracker.update_state(gen_B_loss)
        self.disc_X_loss_tracker.update_state(disc_X_loss)
        self.disc_Y_loss_tracker.update_state(disc_Y_loss)
        self.MSE.update_state(real_y, fake_y)

        return {
            "loss_A": self.gen_A_loss_tracker.result(),
            "loss_B": self.gen_B_loss_tracker.result(),
            "loss_X": self.disc_X_loss_tracker.result(),
            "loss_Y": self.disc_Y_loss_tracker.result(),
            "MSE": self.MSE.result(),
        }

    @ tf.function
    def predict_step(self, data):
        real_x, real_y = data

        # x -> fake_y
        fake_y = self.gen_A(real_x, training=False)

        return fake_y, real_x, real_y

    @ property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch or at the start of
        # `evaluate()`.
        # Without this property, you have to call `reset_states()` yourself
        # at the time of your choosing.
        return [
            self.gen_A_loss_tracker,
            self.gen_B_loss_tracker,
            self.disc_X_loss_tracker,
            self.disc_Y_loss_tracker,
            self.MSE,
        ]

    @ staticmethod
    def show_report_handler(prediction,
                            export_report=None,
                            report_labels=None):
        if report_labels is None:
            report_labels = ["Processed", "Input", "Target"]
        else:
            if not isinstance(report_labels, list) or len(report_labels) != 3:
                raise ValueError(
                    "'report_labels' must be a 3-elements list.")

        # TODO To be implemented
        return

class Generator_Exporter():
    def __init__(self):
        return

    def export_generator(self, checkpoint_filepath, lambda_cycle=10.0, lambda_identity=5.0, lambda_adv=1.0):
        checkpoint_filepath = os.path.abspath(checkpoint_filepath)

        # Parse model parameter from checkpoint folder
        config_raw = "__".join(checkpoint_filepath.split(os.sep)[-2:])
        generator_config_raw = config_raw.split("__")[0].replace("G", "")
        discriminator_config_raw = config_raw.split("__")[1].replace("D", "")

        generator_config = self.__parse_gen_config__(generator_config_raw)
        discriminator_config = self.__parse_disc_config__(
            discriminator_config_raw)

        # Allocate the proper base models
        genA = genB = Inception_Unet(signal_len=generator_config["signal_len"],
                                     n_channels=generator_config["n_channels"],
                                     n_layers=generator_config["n_layers"],
                                     filters=generator_config["filters"],
                                     inception_layers=generator_config["inception_layers"],
                                     inception_receptive_fields=generator_config["inception_receptive_fields"],
                                     traditional_block_kernel=generator_config["traditional_block_kernel"],
                                     recurrent_skip=generator_config["recurrent_skip"],
                                     recurrent_exit=generator_config["recurrent_exit"],
                                     exit_recurrent_units=generator_config["exit_recurrent_units"],
                                     variable_kernel_size=generator_config["variable_kernel_size"])

        discX = discY = Discriminator(signal_len=discriminator_config["signal_len"],
                                      n_channels=discriminator_config["n_channels"],
                                      n_layers=discriminator_config["n_layers"],
                                      filters=discriminator_config["filters"],
                                      kernel_size=discriminator_config["kernel_size"],
                                      variable_kernel_size=discriminator_config["variable_kernel_size"])

        loaded_model = cycleGAN(generator_A=genA,
                                generator_B=genB,
                                discriminator_X=discX,
                                discriminator_Y=discY,
                                lambda_cycle=lambda_cycle,
                                lambda_identity=lambda_identity,
                                lambda_adv=lambda_adv,
                                restore_filepath=checkpoint_filepath)

        # Extract the generator
        loaded_generator = loaded_model.gen_A

        loaded_generator.compile()

        return loaded_generator, loaded_model

    @ staticmethod
    def __parse_gen_config__(config_raw):
        # parse Generator configuration
        config_dict = {"signal_len": None,
                       "n_channels": None,
                       "n_layers": None,
                       "filters": None,
                       "inception_layers": 0,
                       "inception_receptive_fields": [],
                       "traditional_block_kernel": None,
                       "recurrent_skip": False,
                       "recurrent_exit": False,
                       "exit_recurrent_units": None,
                       "variable_kernel_size": False,
                       }
        temp_param = config_raw.split("_")
        for param in temp_param:
            if "x" in param:
                config_dict["signal_len"] = int(param.split("x")[0])
                config_dict["n_channels"] = int(param.split("x")[1])
            elif "l" in param:
                config_dict["n_layers"] = int(param.replace("l", ""))
            elif "f" in param:
                config_dict["filters"] = int(param.replace("f", ""))
            elif "inc" in param:
                config_dict["inception_layers"] = int(param.replace("inc", ""))
            elif "-" in param:
                config_dict["inception_receptive_fields"] = [
                    int(x) for x in param.split("-")]
            elif "k" in param:
                config_dict["traditional_block_kernel"] = int(
                    param.replace("k", ""))
            elif "S" in param:
                config_dict["recurrent_skip"] = True
            elif "E" in param:
                config_dict["recurrent_exit"] = True
                config_dict["exit_recurrent_units"] = int(
                    param.replace("E", ""))
            elif "Var" in param:
                config_dict["variable_kernel_size"] = True

        print("Generator config:")
        for key, val in config_dict.items():
            print(f"\t{key:.<30}{val}")

        return config_dict

    @ staticmethod
    def __parse_disc_config__(config_raw):
        # parse Discriminator configuration
        config_dict = {"signal_len": None,
                       "n_channels": None,
                       "n_layers": None,
                       "filters": None,
                       "kernel_size": None,
                       "variable_kernel_size": False,
                       }
        temp_param = config_raw.split("_")
        print(temp_param)
        for param in temp_param:
            if "x" in param:
                config_dict["signal_len"] = int(param.split("x")[0])
                config_dict["n_channels"] = int(param.split("x")[1])
            elif "l" in param:
                config_dict["n_layers"] = int(param.replace("l", ""))
            elif "f" in param:
                config_dict["filters"] = int(param.replace("f", ""))
            elif "k" in param:
                config_dict["kernel_size"] = int(param.replace("k", ""))
            elif "Var" in param:
                config_dict["variable_kernel_size"] = True
        print("Discriminator config:")
        for key, val in config_dict.items():
            print(f"\t{key:.<30}{val}")

        return config_dict