import torch

from fairseq.data import FileAudioDataset, EpochBatchIterator
from fairseq.data.audio.raw_audio_dataset import LogMelAudioDataset, apply_mv_norm
from fairseq.models.wav2vec import ConformerEncoderLayer
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # print('Raw Waveform')
    # data = FileAudioDataset(
    #     manifest_path='/checkpoint/anuroops/data/libris/proc/valid.tsv',
    #     sample_rate=16000,
    # )
    # print(len(data))
    # print(data[0])
    # print(data[0]['source'].dtype, data[0]['source'].shape)

    # batch_sampler = data.batch_by_size(data.ordered_indices())
    # loader = EpochBatchIterator(
    #     data, collate_fn=data.collater, batch_sampler=batch_sampler, num_workers=1
    # )
    # print(loader.n)
    # for batch in loader.next_epoch_itr(shuffle=False):
    #     print(batch)
    #     print(batch['id'].shape, batch['net_input']['source'].shape)
    #     break

    print('Log MEL')
    data1 = LogMelAudioDataset(
        manifest_path='/checkpoint/anuroops/data/libris/proc/train.tsv',
        sample_rate=16000,
        max_sample_size=250000,
        min_sample_size=32000,
    )
    for i in range(1):
        print(len(data1))
        print(data1[40])
        src = data1[40]['source']
        print(src.dtype, src.shape)
        print(src.min(), src.max())
        plt.imsave(f'imgs/spectrogram.{i}.png', src.numpy().T)
    data2 = LogMelAudioDataset(
        manifest_path='/checkpoint/anuroops/data/libris/proc/train.tsv',
        sample_rate=16000,
        max_sample_size=250000,
        min_sample_size=32000,
        specaug_prob=0.4,
    )
    for i in range(10):
        print(len(data2))
        print(data2[40])
        src = data2[40]['source']
        print(src.dtype, src.shape)
        print(src.min(), src.max())
        plt.imsave(f'imgs/spectrogram.{i}.specaug.png', src.numpy().T)

    # batch_sampler = data.batch_by_size(data.ordered_indices())
    # loader = EpochBatchIterator(
    #     data, collate_fn=data.collater,
    #     batch_sampler=batch_sampler, num_workers=6,
    # )
    # print(len(loader))
    # for batch in loader.next_epoch_itr():
    #     print(batch)
    #     print(batch['id'].shape, batch['net_input']['source'].shape)
    #
    import python_speech_features
    fl = '/datasets01_101/librispeech/021419/data/dev-clean/000001929.flac'
    import soundfile as sf

    wav, curr_sample_rate = sf.read(fl)
    print(wav.shape, curr_sample_rate)
    logmel = python_speech_features.base.logfbank(
        wav, curr_sample_rate,
        winlen=25 / 1000.,
        winstep=10 / 1000.,
        nfilt=80,
        lowfreq=20.,
        wintype='povey',
        dither=0.
    )
    print(logmel.shape)
    plt.imsave(f'imgs/spectrogram.png', logmel.T)
    logmel_norm = apply_mv_norm(torch.tensor(logmel)).numpy()
    plt.imsave(f'imgs/spectrogram_norm.png', logmel_norm.T)

