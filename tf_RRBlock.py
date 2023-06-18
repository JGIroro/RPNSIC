

import tensorflow as tf
import tensorflow_compression as tfc

def gumbel_softmax(x, dim, tau):
    gumbels = tf.random_uniform(tf.shape(x))
    while tf.greater(tf.reduce_mean(tf.cast(tf.equal(gumbels, 0),dtype = 'int32')), 0):
        gumbels = tf.random_uniform(tf.shape(x))

    gumbels = -tf.log((-tf.log(gumbels)))
    gumbels = (x + gumbels) / tau
    x = tf.nn.softmax(gumbels)

    return x

class SMB((tf.keras.layers.Layer)):
    def __init__(self, in_channels, out_channels, n_layers, training = True, *args, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        self.tau = 1
        self.relu = tf.keras.layers.ReLU()    # self.relu = nn.ReLU(True)  tf中找不到inplace平替

        self.training = training
        super(SMB, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        # channels mask
        self.ch_mask = tf.Variable(tf.random_uniform((1, self.n_layers, 2, self.out_channels)))
        # self.ch_mask = nn.Parameter(torch.rand(1, out_channels, n_layers, 2))  Parameter也找不到平替
        
        # body
        # body = []
        # body.append(tf.keras.layers.Conv2DTranspose(self.out_channels, kernel_size=3, strides=(1, 1), padding="valid", use_bias=False))
        # for _ in range(self.n_layers-1):
        #     body.append(tf.keras.layers.Conv2DTranspose(self.out_channels, kernel_size=3, stride=(1, 1), padding="valid", use_bias=False))
        # self.body = tf.keras.Sequential(*body)

        self.body = [
            tfc.SignalConv2D(
                self.out_channels, (3, 3), name="layer_0", corr=True, strides_down=1,
                padding="same_zeros", use_bias=False,
                activation=None),
            tfc.SignalConv2D(
                self.out_channels, (3, 3), name="layer_1", corr=True, strides_down=1,
                padding="same_zeros", use_bias=False,
                activation=None),
            tfc.SignalConv2D(
                self.out_channels, (3, 3), name="layer_2", corr=True, strides_down=1,
                padding="same_zeros", use_bias=False,
                activation=None),
            tfc.SignalConv2D(
                self.out_channels, (3, 3), name="layer_3", corr=True, strides_down=1,
                padding="same_zeros", use_bias=False,
                activation=None),
        ]

        # collect
        self.collect = [tf.keras.layers.Conv2DTranspose(self.out_channels, kernel_size=1, strides=(1, 1), padding="same",
                                    name = 'collect', activation=None),]

        super(SMB, self).build(input_shape)

    def _update_tau(self, tau):
        self.tau = tau

    def _prepare(self):
        # channel mask
        ch_mask = tf.round(tf.nn.softmax(self.ch_mask, dim = 1))
        self.ch_mask_round = ch_mask

        # number of channels
        self.d_in_num = []
        self.s_in_num = []
        self.d_out_num = []
        self.s_out_num = []

        for i in range(self.n_layers):
            if i == 0:
                self.d_in_num.append(self.in_channels)
                self.s_in_num.append(0)
                self.d_out_num.append(int(ch_mask[0, i, 0, :].sum(0)))
                self.s_out_num.append(int(ch_mask[0, i, 1, :].sum(0)))
            else:
                self.d_in_num.append(int(ch_mask[0, i-1, 0, :].sum(0)))
                self.s_in_num.append(int(ch_mask[0, i-1, 1, :].sum(0)))
                self.d_out_num.append(int(ch_mask[0, i, 0, :].sum(0)))
                self.s_out_num.append(int(ch_mask[0, i, 1, :].sum(0)))

        # kernel split
        kernel_d2d = []
        kernel_d2s = []
        kernel_s = []

        for i in range(self.n_layers):
            if i == 0:
                kernel_s.append([])
                if self.d_out_num[i] > 0:
                    kernel_d2d.append(tf.reshape(self.body[i].get_weights()[ch_mask[0, i, 0, :]==1, ...], [-1, self.d_out_num[i]]))
                else:
                    kernel_d2d.append([])
                if self.s_out_num[i] > 0:
                    kernel_d2s.append(tf.reshape(self.body[i].get_weights()[ch_mask[0, i, 1, :]==1, ...], [-1, self.s_out_num[i]]))
                else:
                    kernel_d2s.append([])
            else:
                if self.d_in_num[i] > 0 and self.d_out_num[i] > 0:
                    kernel_d2d.append(tf.reshape(
                        self.body[i].get_weights()[ch_mask[0, i, 0, :] == 1, ...][:, ch_mask[0, i-1, 0, :] == 1, ...], [-1, self.d_out_num[i]]))
                else:
                    kernel_d2d.append([])
                if self.d_in_num[i] > 0 and self.s_out_num[i] > 0:
                    kernel_d2s.append(tf.reshape(
                        self.body[i].get_weights()[ch_mask[0, i, 1, :] == 1, ...][:, ch_mask[0, i-1, 0, :] == 1, ...], [-1, self.s_out_num[i]]))
                else:
                    kernel_d2s.append([])
                if self.s_in_num[i] > 0:
                    kernel_s.append(tf.reshape(tf.concat((
                        self.body[i].get_weights()[ch_mask[0, i, 0, :] == 1, ...][:, ch_mask[0, i - 1, 1, :] == 1, ...],
                        self.body[i].get_weights()[ch_mask[0, i, 1, :] == 1, ...][:, ch_mask[0, i - 1, 1, :] == 1, ...]),
                        0), [-1, self.d_out_num[i]+self.s_out_num[i]]))
                else:
                    kernel_s.append([])

        # the last 1x1 conv
        ch_mask = tf.reshape(tf.transpose(ch_mask[..., 0], perm=[1, 0]), [-1, 2])
        self.d_in_num.append(int(tf.reduce_sum(ch_mask[0, :], 0)))
        self.s_in_num.append(int(tf.reduce_sum(ch_mask[1, :], 0)))
        self.d_out_num.append(self.out_channels)
        self.s_out_num.append(0)

        kernel_d2d.append(tf.squeeze(self.collect.get_weights()[:, ..., ch_mask[..., 0] == 1]))
        kernel_d2s.append([])
        kernel_s.append(tf.squeeze(self.collect.get_weights()[:, ..., ch_mask[..., 1] == 1]))

        self.kernel_d2d = kernel_d2d
        self.kernel_d2s = kernel_d2s
        self.kernel_s = kernel_s
        self.bias = self.collect.bias

    def _generate_indices(self):
        A = tf.cast(tf.reshape(tf.range(3), [-1, 1, 1]), tf.int64)
        tmp = tf.squeeze(self.spa_mask)
        mask_indices = tf.where(tf.not_equal(tmp, 0))     #torch.nonzero(self.spa_mask.squeeze())

        # indices: dense to sparse (1x1)
        self.h_idx_1x1 = mask_indices[:, 0]
        self.w_idx_1x1 = mask_indices[:, 1]

        # indices: dense to sparse (3x3)
        mask_indices_repeat = tf.tile(tf.expand_dims(mask_indices, 0), [1, 1, 3]) + A

        self.h_idx_3x3 = tf.reshape(tf.tile(mask_indices_repeat[..., 0], [1, 3]), [-1])
        self.w_idx_3x3 = tf.reshape(tf.tile(mask_indices_repeat[..., 1], [3, 1]), [-1])
        
        # indices: sparse to sparse (3x3)
        indices = tf.reshape(tf.range(tf.to_float(tf.shape(mask_indices)[0])), [-1, 1]) + 1

        # indic = [self.h_idx_1x1, self.w_idx_1x1]
        # valu = [indices]
        # shape = tf.cast(tf.shape(self.spa_mask), tf.int32)
        # delta = tf.SparseTensor(indic, valu, shape)
        # self.spa_mask = self.spa_mask + tf.sparse_tensor_to_dense(delta)        
        # a = tf.Variable(tf.zeros([1,5,5,2]))
        # a = a + self.spa_mask
        # a[0, self.h_idx_1x1, self.w_idx_1x1, 0].assign(indices)
        # # self.spa_mask = a
        # tmp_0 = self.spa_mask[1:, :, :, 1:] 
        # tmp_1 = self.spa_mask[1:, :, :, 1:] 
        # tmp_2 = self.spa_mask[1:, :, :, 1:] 
        # Image = tf.stack([tmp_0,tmp_1,tmp_2],2)

        self.spa_mask[0, self.h_idx_1x1, self.w_idx_1x1, 0] = indices

        self.idx_s2s = tf.reshape(tf.pad(self.spa_mask, [1, 1, 1, 1])[0, self.h_idx_3x3, self.w_idx_3x3, :], [9, -1]).long()

    def _mask_select(self, x, k):
        if k == 1:
            return x[0, self.h_idx_1x1, self.w_idx_1x1, :]
        if k == 3:
            return tf.reshape(tf.pad(x, [1, 1, 1, 1])[0, self.h_idx_3x3, self.w_idx_3x3, :], [9 * x.size(1), -1])

    def _sparse_conv(self, fea_dense, fea_sparse, k, index):
        '''
        :param fea_dense: (B, C_d, H, W)
        :param fea_sparse: (C_s, N)
        :param k: kernel size
        :param index: layer index
        '''
        # dense input
        if self.d_in_num[index] > 0:
            if self.d_out_num[index] > 0:
                # dense to dense
                if k > 1:
                    fea_col = tf.squeeze(tf.extract_image_patches(fea_dense, k, stride=1, padding=(k-1) // 2), 0)   # fea_col = F.unfold(fea_dense, k, stride=1, padding=(k-1) // 2).squeeze(0)
                    fea_d2d = tf.matmul(tf.reshape(self.kernel_d2d[index], [-1, self.d_out_num[index]]), fea_col)
                    fea_d2d = tf.reshape(fea_d2d, [1, fea_dense.size(2), fea_dense.size(3), self.d_out_num[index]])
                else:
                    fea_col = tf.reshape(fea_dense, [self.d_in_num[index], -1])
                    fea_d2d = tf.matmul(tf.reshape(self.kernel_d2d[index], [-1, self.d_out_num[index]]), fea_col)
                    fea_d2d = tf.reshape(fea_d2d, [1, fea_dense.size(2), fea_dense.size(3), self.d_out_num[index]])

            if self.s_out_num[index] > 0:
                # dense to sparse
                fea_d2s = tf.matmul(self.kernel_d2s[index], self._mask_select(fea_dense, k))

        # sparse input
        if self.s_in_num[index] > 0:
            # sparse to dense & sparse
            if k == 1:
                fea_s2ds = tf.matmul(self.kernel_s[index], fea_sparse)
            else:
                fea_s2ds = tf.matmul(self.kernel_s[index], tf.reshape(tf.pad(fea_sparse, [1,0,0,0])[:, self.idx_s2s], [self.s_in_num[index] * k * k, -1]))

        # fusion
        if self.d_out_num[index] > 0:
            if self.d_in_num[index] > 0:
                if self.s_in_num[index] > 0:
                    fea_d2d[0, self.h_idx_1x1, self.w_idx_1x1, :] += fea_s2ds[:, :self.d_out_num[index]]
                    fea_d = fea_d2d
                else:
                    fea_d = fea_d2d
            else:
                fea_d = tf.tile(tf.zeros_like(self.spa_mask), [1, 1, 1, self.d_out_num[index]])
                fea_d[0, self.h_idx_1x1, self.w_idx_1x1, :] = fea_s2ds[:, :self.d_out_num[index]:]
        else:
            fea_d = None

        if self.s_out_num[index] > 0:
            if self.d_in_num[index] > 0:
                if self.s_in_num[index] > 0:
                    fea_s = fea_d2s + fea_s2ds[:,  -self.s_out_num[index]:]
                else:
                    fea_s = fea_d2s
            else:
                fea_s = fea_s2ds[:, -self.s_out_num[index]:]
        else:
            fea_s = None

        # add bias (bias is only used in the last 1x1 conv in our SMB for simplicity)
        if index == 4:
            fea_d += self.bias.view(1, 1, 1, -1)

        return fea_d, fea_s

    def call(self, x):
        '''
        :param x: [x[0], x[1]]
        x[0]: input feature torch (B, C ,H, W) ; tf B H W C
        x[1]: spatial mask (B, 1, H, W) ; tf B H W 1
        '''
        if self.training:
            spa_mask = x[1]
            ch_mask = gumbel_softmax(self.ch_mask, 2, self.tau)

            out = []
            fea = x[0]
            for i in range(self.n_layers):
                if i == 0:
                    fea = self.body[i](fea)
                    fea = fea * ch_mask[:, i:i + 1, 1:, :] * spa_mask + fea * ch_mask[:, i:i + 1, :1, :]
                    # fea = fea * spa_mask
                    #* spa_mask + fea * ch_mask[:, i:i + 1, :1, :]
                else:
                    fea_d = self.body[i](fea * ch_mask[:, i - 1:i, :1, :])
                    fea_s = self.body[i](fea * ch_mask[:, i - 1:i, 1:, :])
                    fea = fea_d * ch_mask[:, i:i + 1, 1:, :] * spa_mask + fea_d * ch_mask[:, i:i + 1, :1, :] + \
                          fea_s * ch_mask[:, i:i + 1, 1:, :] * spa_mask + fea_s * ch_mask[:, i:i + 1, :1, :] * spa_mask
                fea = self.relu(fea)
                out.append(fea)
                # aaa = out+x[0]
            tmp = tf.concat(out, -1)
            out = self.collect[0](tmp)
            

            return out, ch_mask

        if not self.training:
            self.spa_mask = x[1]

            # generate indices
            self._generate_indices()

            # sparse conv
            fea_d = x[0]
            fea_s = None
            fea_dense = []
            fea_sparse = []
            for i in range(self.n_layers):
                fea_d, fea_s = self._sparse_conv(fea_d, fea_s, k=3, index=i)
                if fea_d is not None:
                    fea_dense.append(self.relu(fea_d))
                if fea_s is not None:
                    fea_sparse.append(self.relu(fea_s))

            # 1x1 conv
            fea_dense = tf.concat(fea_dense, 1)
            fea_sparse = tf.concat(fea_sparse, 0)
            out, _ = self._sparse_conv(fea_dense, fea_sparse, k=1, index=self.n_layers)

            return out


class step0_SMB((tf.keras.layers.Layer)):
    def __init__(self, in_channels, out_channels, n_layers, training = True, *args, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        self.tau = 1
        self.relu = tf.keras.layers.ReLU()    # self.relu = nn.ReLU(True)  tf中找不到inplace平替

        self.training = training
        super(step0_SMB, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        # channels mask
        self.ch_mask = tf.Variable(tf.random_uniform((1, self.n_layers, 2, self.out_channels)))
        # self.ch_mask = nn.Parameter(torch.rand(1, out_channels, n_layers, 2))  Parameter也找不到平替
        
        # body
        # body = []
        # body.append(tf.keras.layers.Conv2DTranspose(self.out_channels, kernel_size=3, strides=(1, 1), padding="valid", use_bias=False))
        # for _ in range(self.n_layers-1):
        #     body.append(tf.keras.layers.Conv2DTranspose(self.out_channels, kernel_size=3, stride=(1, 1), padding="valid", use_bias=False))
        # self.body = tf.keras.Sequential(*body)

        self.body = [
            tfc.SignalConv2D(
                self.out_channels, (3, 3), name="layer_0", corr=False, strides_down=1,
                padding="same_zeros", use_bias=False,
                activation=None),
            tfc.SignalConv2D(
                self.out_channels, (3, 3), name="layer_1", corr=False, strides_down=1,
                padding="same_zeros", use_bias=False,
                activation=None),
            tfc.SignalConv2D(
                self.out_channels, (3, 3), name="layer_2", corr=False, strides_down=1,
                padding="same_zeros", use_bias=False,
                activation=None),
            tfc.SignalConv2D(
                self.out_channels, (3, 3), name="layer_3", corr=False, strides_down=1,
                padding="same_zeros", use_bias=False,
                activation=None),
        ]

        # collect
        self.collect = [tf.keras.layers.Conv2DTranspose(self.out_channels, kernel_size=1, strides=(1, 1), padding="same",
                                    name = 'collect', activation=None),]

        super(step0_SMB, self).build(input_shape)

    def _update_tau(self, tau):
        self.tau = tau

    def _prepare(self):
        # channel mask
        ch_mask = tf.round(tf.nn.softmax(self.ch_mask, dim = 1))
        self.ch_mask_round = ch_mask

        # number of channels
        self.d_in_num = []
        self.s_in_num = []
        self.d_out_num = []
        self.s_out_num = []

        for i in range(self.n_layers):
            if i == 0:
                self.d_in_num.append(self.in_channels)
                self.s_in_num.append(0)
                self.d_out_num.append(int(ch_mask[0, i, 0, :].sum(0)))
                self.s_out_num.append(int(ch_mask[0, i, 1, :].sum(0)))
            else:
                self.d_in_num.append(int(ch_mask[0, i-1, 0, :].sum(0)))
                self.s_in_num.append(int(ch_mask[0, i-1, 1, :].sum(0)))
                self.d_out_num.append(int(ch_mask[0, i, 0, :].sum(0)))
                self.s_out_num.append(int(ch_mask[0, i, 1, :].sum(0)))

        # kernel split
        kernel_d2d = []
        kernel_d2s = []
        kernel_s = []

        for i in range(self.n_layers):
            if i == 0:
                kernel_s.append([])
                if self.d_out_num[i] > 0:
                    kernel_d2d.append(tf.reshape(self.body[i].get_weights()[ch_mask[0, i, 0, :]==1, ...], [-1, self.d_out_num[i]]))
                else:
                    kernel_d2d.append([])
                if self.s_out_num[i] > 0:
                    kernel_d2s.append(tf.reshape(self.body[i].get_weights()[ch_mask[0, i, 1, :]==1, ...], [-1, self.s_out_num[i]]))
                else:
                    kernel_d2s.append([])
            else:
                if self.d_in_num[i] > 0 and self.d_out_num[i] > 0:
                    kernel_d2d.append(tf.reshape(
                        self.body[i].get_weights()[ch_mask[0, i, 0, :] == 1, ...][:, ch_mask[0, i-1, 0, :] == 1, ...], [-1, self.d_out_num[i]]))
                else:
                    kernel_d2d.append([])
                if self.d_in_num[i] > 0 and self.s_out_num[i] > 0:
                    kernel_d2s.append(tf.reshape(
                        self.body[i].get_weights()[ch_mask[0, i, 1, :] == 1, ...][:, ch_mask[0, i-1, 0, :] == 1, ...], [-1, self.s_out_num[i]]))
                else:
                    kernel_d2s.append([])
                if self.s_in_num[i] > 0:
                    kernel_s.append(tf.reshape(tf.concat((
                        self.body[i].get_weights()[ch_mask[0, i, 0, :] == 1, ...][:, ch_mask[0, i - 1, 1, :] == 1, ...],
                        self.body[i].get_weights()[ch_mask[0, i, 1, :] == 1, ...][:, ch_mask[0, i - 1, 1, :] == 1, ...]),
                        0), [-1, self.d_out_num[i]+self.s_out_num[i]]))
                else:
                    kernel_s.append([])

        # the last 1x1 conv
        ch_mask = tf.reshape(tf.transpose(ch_mask[..., 0], perm=[1, 0]), [-1, 2])
        self.d_in_num.append(int(tf.reduce_sum(ch_mask[0, :], 0)))
        self.s_in_num.append(int(tf.reduce_sum(ch_mask[1, :], 0)))
        self.d_out_num.append(self.out_channels)
        self.s_out_num.append(0)

        kernel_d2d.append(tf.squeeze(self.collect.get_weights()[:, ..., ch_mask[..., 0] == 1]))
        kernel_d2s.append([])
        kernel_s.append(tf.squeeze(self.collect.get_weights()[:, ..., ch_mask[..., 1] == 1]))

        self.kernel_d2d = kernel_d2d
        self.kernel_d2s = kernel_d2s
        self.kernel_s = kernel_s
        self.bias = self.collect.bias

    def _generate_indices(self):
        A = tf.cast(tf.reshape(tf.range(3), [-1, 1, 1]), tf.int64)
        tmp = tf.squeeze(self.spa_mask)
        mask_indices = tf.where(tf.not_equal(tmp, 0))     #torch.nonzero(self.spa_mask.squeeze())

        # indices: dense to sparse (1x1)
        self.h_idx_1x1 = mask_indices[:, 0]
        self.w_idx_1x1 = mask_indices[:, 1]

        # indices: dense to sparse (3x3)
        mask_indices_repeat = tf.tile(tf.expand_dims(mask_indices, 0), [1, 1, 3]) + A

        self.h_idx_3x3 = tf.reshape(tf.tile(mask_indices_repeat[..., 0], [1, 3]), [-1])
        self.w_idx_3x3 = tf.reshape(tf.tile(mask_indices_repeat[..., 1], [3, 1]), [-1])
        
        # indices: sparse to sparse (3x3)
        indices = tf.reshape(tf.range(tf.to_float(tf.shape(mask_indices)[0])), [-1, 1]) + 1

        # indic = [self.h_idx_1x1, self.w_idx_1x1]
        # valu = [indices]
        # shape = tf.cast(tf.shape(self.spa_mask), tf.int32)
        # delta = tf.SparseTensor(indic, valu, shape)
        # self.spa_mask = self.spa_mask + tf.sparse_tensor_to_dense(delta)        
        # a = tf.Variable(tf.zeros([1,5,5,2]))
        # a = a + self.spa_mask
        # a[0, self.h_idx_1x1, self.w_idx_1x1, 0].assign(indices)
        # # self.spa_mask = a
        # tmp_0 = self.spa_mask[1:, :, :, 1:] 
        # tmp_1 = self.spa_mask[1:, :, :, 1:] 
        # tmp_2 = self.spa_mask[1:, :, :, 1:] 
        # Image = tf.stack([tmp_0,tmp_1,tmp_2],2)

        self.spa_mask[0, self.h_idx_1x1, self.w_idx_1x1, 0] = indices

        self.idx_s2s = tf.reshape(tf.pad(self.spa_mask, [1, 1, 1, 1])[0, self.h_idx_3x3, self.w_idx_3x3, :], [9, -1]).long()

    def _mask_select(self, x, k):
        if k == 1:
            return x[0, self.h_idx_1x1, self.w_idx_1x1, :]
        if k == 3:
            return tf.reshape(tf.pad(x, [1, 1, 1, 1])[0, self.h_idx_3x3, self.w_idx_3x3, :], [9 * x.size(1), -1])

    def _sparse_conv(self, fea_dense, fea_sparse, k, index):
        '''
        :param fea_dense: (B, C_d, H, W)
        :param fea_sparse: (C_s, N)
        :param k: kernel size
        :param index: layer index
        '''
        # dense input
        if self.d_in_num[index] > 0:
            if self.d_out_num[index] > 0:
                # dense to dense
                if k > 1:
                    fea_col = tf.squeeze(tf.extract_image_patches(fea_dense, k, stride=1, padding=(k-1) // 2), 0)   # fea_col = F.unfold(fea_dense, k, stride=1, padding=(k-1) // 2).squeeze(0)
                    fea_d2d = tf.matmul(tf.reshape(self.kernel_d2d[index], [-1, self.d_out_num[index]]), fea_col)
                    fea_d2d = tf.reshape(fea_d2d, [1, fea_dense.size(2), fea_dense.size(3), self.d_out_num[index]])
                else:
                    fea_col = tf.reshape(fea_dense, [self.d_in_num[index], -1])
                    fea_d2d = tf.matmul(tf.reshape(self.kernel_d2d[index], [-1, self.d_out_num[index]]), fea_col)
                    fea_d2d = tf.reshape(fea_d2d, [1, fea_dense.size(2), fea_dense.size(3), self.d_out_num[index]])

            if self.s_out_num[index] > 0:
                # dense to sparse
                fea_d2s = tf.matmul(self.kernel_d2s[index], self._mask_select(fea_dense, k))

        # sparse input
        if self.s_in_num[index] > 0:
            # sparse to dense & sparse
            if k == 1:
                fea_s2ds = tf.matmul(self.kernel_s[index], fea_sparse)
            else:
                fea_s2ds = tf.matmul(self.kernel_s[index], tf.reshape(tf.pad(fea_sparse, [1,0,0,0])[:, self.idx_s2s], [self.s_in_num[index] * k * k, -1]))

        # fusion
        if self.d_out_num[index] > 0:
            if self.d_in_num[index] > 0:
                if self.s_in_num[index] > 0:
                    fea_d2d[0, self.h_idx_1x1, self.w_idx_1x1, :] += fea_s2ds[:, :self.d_out_num[index]]
                    fea_d = fea_d2d
                else:
                    fea_d = fea_d2d
            else:
                fea_d = tf.tile(tf.zeros_like(self.spa_mask), [1, 1, 1, self.d_out_num[index]])
                fea_d[0, self.h_idx_1x1, self.w_idx_1x1, :] = fea_s2ds[:, :self.d_out_num[index]:]
        else:
            fea_d = None

        if self.s_out_num[index] > 0:
            if self.d_in_num[index] > 0:
                if self.s_in_num[index] > 0:
                    fea_s = fea_d2s + fea_s2ds[:,  -self.s_out_num[index]:]
                else:
                    fea_s = fea_d2s
            else:
                fea_s = fea_s2ds[:, -self.s_out_num[index]:]
        else:
            fea_s = None

        # add bias (bias is only used in the last 1x1 conv in our SMB for simplicity)
        if index == 4:
            fea_d += self.bias.view(1, 1, 1, -1)

        return fea_d, fea_s

    def call(self, x):
        '''
        :param x: [x[0], x[1]]
        x[0]: input feature torch (B, C ,H, W) ; tf B H W C
        x[1]: spatial mask (B, 1, H, W) ; tf B H W 1
        '''
        if self.training:
            spa_mask = x[1]
            ch_mask = gumbel_softmax(self.ch_mask, 2, self.tau)

            out = []
            fea = x[0]
            for i in range(self.n_layers):
                if i == 0:
                    fea = self.body[i](fea)
                    fea = fea * ch_mask[:, i:i + 1, 1:, :] * spa_mask + fea * ch_mask[:, i:i + 1, :1, :]
                    # fea = fea * spa_mask
                    #* spa_mask + fea * ch_mask[:, i:i + 1, :1, :]
                else:
                    fea_d = self.body[i](fea * ch_mask[:, i - 1:i, :1, :])
                    fea_s = self.body[i](fea * ch_mask[:, i - 1:i, 1:, :])
                    fea = fea_d * ch_mask[:, i:i + 1, 1:, :] * spa_mask + fea_d * ch_mask[:, i:i + 1, :1, :] + \
                          fea_s * ch_mask[:, i:i + 1, 1:, :] * spa_mask + fea_s * ch_mask[:, i:i + 1, :1, :] * spa_mask
                fea = self.relu(fea)
                out.append(fea)
                # aaa = out+x[0]
            tmp = tf.concat(out, -1)
            out = self.collect[0](tmp)
            

            return out, ch_mask

        if not self.training:
            self.spa_mask = x[1]

            # generate indices
            self._generate_indices()

            # sparse conv
            fea_d = x[0]
            fea_s = None
            fea_dense = []
            fea_sparse = []
            for i in range(self.n_layers):
                fea_d, fea_s = self._sparse_conv(fea_d, fea_s, k=3, index=i)
                if fea_d is not None:
                    fea_dense.append(self.relu(fea_d))
                if fea_s is not None:
                    fea_sparse.append(self.relu(fea_s))

            # 1x1 conv
            fea_dense = tf.concat(fea_dense, 1)
            fea_sparse = tf.concat(fea_sparse, 0)
            out, _ = self._sparse_conv(fea_dense, fea_sparse, k=1, index=self.n_layers)

            return out
