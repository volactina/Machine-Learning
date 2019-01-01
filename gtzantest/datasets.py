import os
import tarfile
import urllib
import pickle
import librosa


def fetch_dataset(name, url, datasets_path='datasets'):
    print("==> Fetching dataset '{}', with results:".format(name))
    if not os.path.isdir(datasets_path):
        os.makedirs(datasets_path)

    dataset_path = os.path.join(datasets_path, name)
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)

    archive_path = os.path.join(datasets_path, "{}.tar.gz".format(name))
    if not os.path.isfile(archive_path):
        print("Fetching archive from ''...".format(url))
        urllib.request.urlretrieve(url, archive_path)
    else:
        print("Dataset archive '{}' already exists.".format(name))

    if not os.path.isdir(dataset_path):
        print("Extracting dataset '{}'...".format(name))
        archive = tarfile.open(archive_path)
        archive.extractall(path=dataset_path)
        archive.close()
    else:
        print("Extracted dataset '{}' already exists.".format(name))

    print("==> Fetched dataset '{}'.\n".format(name))


def load_dataset_gtzan(name='gtzan', path='genres', datasets_path='datasets', sampling_rate=22050,
                       load_subset=False, load_object=True):
    print("==> Loading dataset '{}', with results:".format(name))
    if load_subset:
        name = "{}_subset".format(name)

    dataset_object_path = os.path.join(datasets_path, "{}.dataset".format(name))
    if load_object and os.path.isfile(dataset_object_path):
        print("Saved dataset '{}' exists, loading...".format(name))
        dataset = pickle.load(open(dataset_object_path, 'rb'))
    else:
        dataset_path = os.path.join(datasets_path, name, path)
        if not os.path.isdir(dataset_path):
            print("Failed to load: dataset '{}' does not exist.".format(name))
            return None

        GENRES_LIST = ["blues",
                       "classical",
                       "country",
                       "disco",
                       "hiphop",
                       "jazz",
                       "metal",
                       "pop",
                       "reggae",
                       "rock"]

        dataset = []
        for genre in GENRES_LIST:
            audio_files = [os.path.join(d, f)
                           for d, dirs, files in os.walk(os.path.join(dataset_path, genre))
                           for f in files if f.endswith(".au")]
            for filename in audio_files:
                waveform, sampling_rate = librosa.load(
                    filename, sr=sampling_rate)
                sample = {'waveform': waveform,
                          'sampling_rate': sampling_rate,
                          'genre': genre}
                dataset.append(sample)

        print("Saving dataset '{}' to '{}'...".format(name, dataset_object_path))
        pickle.dump(dataset, open(dataset_object_path, 'wb'))
        print("Saved dataset '{}'.".format(name))

    print("==> Loaded dataset '{}'.\n".format(name))
    return dataset