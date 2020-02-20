import h5py
import datetime as dt
import numpy as np

import time as t

# hard coded for now
TIME_OFFSET = dt.timedelta(hours=-6, minutes=4, seconds=9)

class DiveMeta:
    def __init__(self, meta_file):
        with open(meta_file, 'r') as f:
            dives_raw = f.read().split('=') # split dives on '=========='
            dives_raw = [dive for dive in dives_raw if dive] # remove empty strings
        self.dives = []
        for dive_raw in dives_raw:
            self.dives.append(self._parse_dive(dive_raw))

    def _parse_dive(self, dive_str):
        raw_dict = {}
        dive_str = dive_str.replace(' ', '')
        for pair_str in dive_str.split('\n'):
            if ':' not in pair_str:
                continue
            arr = pair_str.split(':')
            if len(arr) <= 1:
                continue
            else:
                raw_dict[arr[0]] =  arr[1]

        # parse date time
        month, day, year = (int(num) for num in raw_dict['date'].split('/'))
        year += 2000 # we are in the 21st century oh yeah
        utc_offset = dt.timedelta(hours=int(raw_dict['time_zone'].replace('UTC', '')))
        # get start time
        hour, minute = (int(num) for num in raw_dict['start_time'].split('/'))
        start_time = dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc)
        start_time -= utc_offset
        # get end time
        hour, minute = (int(num) for num in raw_dict['end_time'].split('/'))
        end_time = dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc)
        end_time -= utc_offset
        # parse gps coordinate
        latitude, longitude = (float(num) for num in raw_dict['gps'].split(','))

        return {
            'start_time': start_time,
            'end_time': end_time,
            'gps': (latitude, longitude),
            'name': raw_dict['folder'],
        }

class FrameGroup:

    def __init__(self, filename, meta):
        self.file = None
        self.file = h5py.File(filename, 'r')
        attr = self.file['Attributes']
        fps = attr['Frame Rate'][0]
        time = self._parse_time(attr['timestamp'][()].decode())
        self.num_frames = attr['num_of_frame'][0]
        self.timestamp = []
        # match with dive
        self.dive = None
        for dive in meta.dives:
            if time >= dive['start_time'] and time <= dive['end_time']:
                self.dive = dive
                break
        assert self.dive is not None, 'Unmatched time at {}'.format(time)
        # calculate precise time for each frame
        for i in range(self.num_frames):
            self.timestamp.append(time + dt.timedelta(microseconds=i * 1e6 / fps))

    def __len__(self):
        return self.num_frames

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def __getitem__(self, key):
        assert isinstance(key, int), 'Only support integer index access!'
        return {
            'image': self.file['Data']['Data'][key],
            'datetime': self.timestamp[key],
            'gps': self.dive['gps'],
            'dive': self.dive['name'],
        }

    def _parse_time(self, time_desc):
        date_str, time_str = time_desc.split('_')
        month, day, year = [int(num) for num in date_str.split('-')]
        hour, minute, second = [int(num) for num in time_str.split('-')]
        utc_time = dt.datetime(
            year, month, day, hour, minute, second, tzinfo=dt.timezone.utc)
        utc_time -= TIME_OFFSET
        return utc_time

