from examples.speech_recognition.data import AsrDataset

from fairseq.data import FileAudioDataset


if __name__ == '__main__':
    data = FileAudioDataset(
        manifest_path='/checkpoint/anuroops/data/libris/proc/valid.tsv',
        sample_rate=16000,
    )
    print(len(data))
    print(data[0])
    print(data[0]['source'].dtype)

    # AsrDataset()
