from fairseq.data import FileAudioDataset, EpochBatchIterator
from fairseq.data.audio.raw_audio_dataset import LogMelAudioDataset

if __name__ == '__main__':
    # print('Raw Waveform')
    # data = FileAudioDataset(
    #     manifest_path='/checkpoint/anuroops/data/libris/proc/valid.tsv',
    #     sample_rate=16000,
    # )
    # print(len(data))
    # print(data[0])
    # print(data[0]['source'].dtype, data[0]['source'].shape)
    #
    # # batch_sampler = data.batch_by_size(data.ordered_indices())
    # # loader = EpochBatchIterator(
    # #     data, collate_fn=data.collater, batch_sampler=batch_sampler, num_workers=1
    # # )
    # # print(loader.n)
    # # for batch in loader.next_epoch_itr(shuffle=False):
    # #     print(batch)
    # #     print(batch['id'].shape, batch['net_input']['source'].shape)
    # #     break
    #
    # print('Log MEL')
    # data = LogMelAudioDataset(
    #     manifest_path='/checkpoint/anuroops/data/libris/proc/train.tsv',
    #     sample_rate=16000,
    #     max_sample_size=250000,
    #     min_sample_size=32000,
    # )
    # print(len(data))
    # print(data[0])
    # src = data[0]['source']
    # print(src.dtype, src.shape)
    # print(src.min(), src.max())
    #
    # batch_sampler = data.batch_by_size(data.ordered_indices())
    # loader = EpochBatchIterator(
    #     data, collate_fn=data.collater,
    #     batch_sampler=batch_sampler, num_workers=6,
    # )
    # print(len(loader))
    # for batch in loader.next_epoch_itr():
    #     print(batch)
    #     print(batch['id'].shape, batch['net_input']['source'].shape)

    # import python_speech_features
    # fl = '/datasets01_101/librispeech/021419/data/dev-clean/000001929.flac'
    # import soundfile as sf
    #
    # wav, curr_sample_rate = sf.read(fl)
    # print(wav.shape, curr_sample_rate)
    # logmel = python_speech_features.base.logfbank(
    #     wav, curr_sample_rate
    # )
    # print(logmel.shape)

    import torch
    import numpy as np

    bins = 80
    framelen = 25
    frameshift = 10

    fl = '/datasets01/librispeech/062419/train-clean-100/103/1240/103-1240-0000.flac'
    import soundfile as sf
    signal, sr = sf.read(fl)

    import torchaudio.compliance.kaldi as kaldi
    fbank = kaldi.fbank(
        torch.tensor(signal).float().unsqueeze(0),
        sample_frequency=sr,
        num_mel_bins=bins,
        frame_length=framelen,
        frame_shift=frameshift,
    )
    fbank = fbank.numpy()
    print(fbank.shape)

    from python_speech_features.base import logfbank

    fbank2 = logfbank(
        signal,
        samplerate=sr,
        winlen=framelen / 1000.,
        winstep=frameshift / 1000.,
        nfilt=bins,
        lowfreq=20.,
        wintype='povey',
        dither=0.
    ).astype(np.float32)
    print(fbank2.shape)

    print(fbank.dtype)
    print(fbank2.dtype)

    print(np.linalg.norm(fbank))
    print(np.linalg.norm(fbank2))

    print(fbank.mean(), fbank.std())
    print(fbank2.mean(), fbank2.std())

    print(fbank.min(), fbank.max())
    print(fbank2.min(), fbank2.max())