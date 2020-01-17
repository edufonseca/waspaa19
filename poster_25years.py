
"""
Figs to illustrate mixup for poster for MTG25 years, where
I presented part of the work for WASPAA19.

This should go in main4.py right after tr_gen_patch is defined (ie the
data loader for the train set).

calls to tr_gen_patch.__getitem__(index_batch)

and in  __getitem__(index_batch) we use:


if self.mixup and self.val_mode is False:
    # features, y_cat = mixup(mode=self.mixup_mode,
    #                         index=index,
    #                         all_patch_indexes=self.indexes,
    #                         batch_size=self.batch_size,
    #                         all_labels=self.labels,
    #                         all_features=self.features,
    #                         n_classes=self.n_classes,
    #                         alpha=self.mixup_alpha,
    #                         )

    # vip deleteme 25 years
    features, y_cat, features1, y_cat1, features2, y_cat2 = mixup(mode=self.mixup_mode,
                                                                      index=index,
                                                                      all_patch_indexes=self.indexes,
                                                                      batch_size=self.batch_size,
                                                                      all_labels=self.labels,
                                                                      all_features=self.features,
                                                                      n_classes=self.n_classes,
                                                                      alpha=self.mixup_alpha,
                                                                      )
and consequently, in mixup() we return:
    return features_out, y_cat_out, _features1, _y_cat1, _features2, _y_cat2



With this, we have access to:
 - the constituents and
 - the mixed up outcome spectrogram
nice to plot it for poster/slides

"""

# vip for MTF 25 years--------------------------------------------------------------
# # features, y_cat, features1, y_cat1, features2, y_cat2
# # --------------------------------sanity check
# index_batch = 0
# index_sample = 0
# batch_features_tr, batch_labels_tr, features1, y_cat1, features2, y_cat2 = tr_gen_patch.__getitem__(index_batch)
# print(batch_features_tr.shape)
# # (batch_size, 1, time, freq)
# print(batch_labels_tr.shape)
# label = np.nonzero(batch_labels_tr[index_sample, :])[0]
# print("Category (nonzero): {}".format([int_to_label[lab] for lab in label]))
# print("Category (max): {}".format(int_to_label[np.argmax(batch_labels_tr[index_sample, :])]))
#
# # watch this code is for poster 25 years, and will be removed in additional unstashed script
# path_pics = 'figs25yearsposter'
#
# # cmap = plt.get_cmap('PiYG')
#
# print('====spectrogram 1 is label {} with id {}'.format(int_to_label[np.argmax(y_cat1[index_sample])],
#                                                         np.argmax(y_cat1[index_sample])))
# plt.figure()
# plt.pcolormesh(np.transpose(features1[index_sample, :, :]))
# plt.xlabel('time [frames]', fontsize=17)
# plt.ylabel('frequency [mel bands]', fontsize=17)
# plt.show()
# plt.savefig(os.path.join(path_pics, 'spec1_.png'), bbox_inches='tight')
#
# print('====spectrogram 2 is label {} with id {}'.format(int_to_label[np.argmax(y_cat2[index_sample])],
#                                                         np.argmax(y_cat2[index_sample])))
# plt.figure()
# # plt.pcolormesh(np.transpose(features1[0, :, :]), shading='gouraud')
# plt.pcolormesh(np.transpose(features2[index_sample, :, :]))
# plt.xlabel('time [frames]', fontsize=17)
# plt.ylabel('frequency [mel bands]', fontsize=17)
# plt.show()
# plt.savefig(os.path.join(path_pics, 'spec2_.png'), bbox_inches='tight')
#
# # print mixup results
# # (batch_size, 1, time, freq) for channels_first
# plt.figure()
# plt.pcolormesh(np.transpose(batch_features_tr[index_sample, 0, :, :]))
# plt.xlabel('time [frames]', fontsize=17)
# plt.ylabel('frequency [mel bands]', fontsize=17)
# plt.show()
# plt.savefig(os.path.join(path_pics, 'spec_mixedup_.png'), bbox_inches='tight')
# vip ------------------------------------- END