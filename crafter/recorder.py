import datetime
import json
import pathlib
import warnings

import imageio
import numpy as np
from framework.visualize.plot import Video


class Recorder:

    def __init__(self, env, directory, helper=None, env_prefix='', save_stats=True, save_video=True, save_episode=True, video_size=(512, 512), log_every_n_episodes=None):
        if directory and save_stats:
            env = StatsRecorder(env, directory, helper, env_prefix, log_every_n_episodes)
        if directory and save_video:
            env = VideoRecorder(env, directory, helper, env_prefix, video_size)
        if directory and save_episode:
            env = EpisodeRecorder(env, directory, helper)
        self._env = env
    
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)


class StatsRecorder:

    def __init__(self, env, directory, helper=None, env_prefix='', log_every_n_episodes=None):
        self._env = env
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(exist_ok=True, parents=True)
        self._file = (self._directory / 'stats.jsonl').open('a')
        self._length = None
        self._reward = None
        self._unlocked = None
        self._stats = None
        self._env_prefix = env_prefix
        self._helper = helper
        if self._helper is not None:
            if log_every_n_episodes is None:
                self._log_every_n_episodes = helper.args.log_every_n_episodes
            else:
                self._log_every_n_episodes = log_every_n_episodes
        self._episode_cnt = 0
        self._logs_cnt = 0
        self._logs_scr = {}
        self._reward_mean = 0
        self._length_mean = 0

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)
    
    def reset(self):
        obs = self._env.reset()
        self._length = 0
        self._reward = 0
        self._unlocked = None
        self._stats = None
        return obs
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._length += 1
        self._reward += info['reward']
        if done:
            self._log_jsonl(info)
            self._episode_cnt += 1
            if self._helper is not None:
                self._log_helper_record(info)  # recorded at every step to match manual scores/rewards
                if self._episode_cnt % self._log_every_n_episodes == 0:
                    self._log_helper(info)
        return obs, reward, done, info
    
    def _save(self):
        self._file.write(json.dumps(self._stats) + '\n')
        self._file.flush()

    def _log_jsonl(self, info):
        self._stats = {'length': self._length, 'reward': round(self._reward, 1)}
        for key, value in info['achievements'].items():
            self._stats[f'achievement_{key}'] = value
        self._save()
    
    def _log_helper_record(self, info):
        if self._logs_cnt == 0:
            for key, value in info['achievements'].items():
                self._logs_scr[f'ach_scr_{key}'] = 0
        self._logs_cnt += 1
        n = self._logs_cnt
        self._reward_mean = self._reward_mean * (n - 1) / n + round(self._reward, 1) / n
        self._length_mean = self._length_mean * (n - 1) / n + self._length / n
        for key, value in info['achievements'].items():
            scr_key = f'ach_scr_{key}'
            self._logs_scr[scr_key] = (n - 1) / n * self._logs_scr[scr_key] + int(value>=1) * 100 / n

    def _log_helper(self, info):
        logs = {
            'length': self._length,
            'length_mean': self._length_mean,
            'reward': round(self._reward, 1),
            'reward_mean': self._reward_mean,
            'episodes': self._episode_cnt,
        }
        for key, value in info['achievements'].items():
            logs[f'ach_cnt_{key}'] = value
            scr_key = f'ach_scr_{key}'
            logs[scr_key] = self._logs_scr[scr_key]
        logs['crf_scr'] = self._crafter_score()
        prefix_logs = {f"{self._env_prefix}/{k}": v for k, v in logs.items()}
        self._helper.log(prefix_logs, step=self._helper.state.step)
    
    def _crafter_score(self):
        # Geometric mean with an offset of 1%.
        with warnings.catch_warnings():  # Empty seeds become NaN.
            warnings.simplefilter('ignore', category=RuntimeWarning)
            percentages = np.array(list(self._logs_scr.values()))
            scores = np.exp(np.nanmean(np.log(1 + percentages))) - 1
        return scores


class VideoRecorder:

    def __init__(self, env, directory, helper, env_prefix, size=(512, 512)):
        if not hasattr(env, 'episode_name'):
            env = EpisodeName(env)
        self._env = env
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(exist_ok=True, parents=True)
        self._size = size
        self._frames = None
        self._env_prefix = env_prefix
        self._helper = helper
    
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)
    
    def reset(self):
        obs = self._env.reset()
        self._frames = [self._env.render(self._size)]
        return obs
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)    
        self._frames.append(self._env.render(self._size))
        if done:
            self._save()
        return obs, reward, done, info
    
    def _save(self):
        # Write to disc
        filename = str(self._directory / (self._env.episode_name + '.mp4'))
        imageio.mimsave(filename, self._frames)
        # Wandb/tensorboard write
        if self._helper:
            logs = {f'{self._env_prefix}/video_render_{self._env.episode_name}': Video(np.array(self._frames, dtype=np.uint8))}
            self._helper.log(logs, step=self._helper.state.step)


class EpisodeRecorder:

    def __init__(self, env, directory, helper):
        if not hasattr(env, 'episode_name'):
            env = EpisodeName(env)
        self._env = env
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(exist_ok=True, parents=True)
        self._episode = None
    
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        self._episode = [{'image': obs}]
        return obs
    
    def step(self, action):
        # Transitions are defined from the environment perspective, meaning that a
        # transition contains the action and the resulting reward and next
        # observation produced by the environment in response to said action.
        obs, reward, done, info = self._env.step(action)
        transition = {
            'action': action, 'image': obs, 'reward': reward, 'done': done,
        }
        for key, value in info.items():
            if key in ('inventory', 'achievements'):
                continue
            transition[key] = value
        for key, value in info['achievements'].items():
            transition[f'achievement_{key}'] = value
        for key, value in info['inventory'].items():
            transition[f'ainventory_{key}'] = value  # TODO: ainventory or inventory?
        self._episode.append(transition)
        if done:
            self._save()
        return obs, reward, done, info
    
    def _save(self):
        filename = str(self._directory / (self._env.episode_name + '.npz'))
        # Fill in zeros for keys missing at the first time step.
        for key, value in self._episode[1].items():
            if key not in self._episode[0]:
                self._episode[0][key] = np.zeros_like(value)
        episode = {
            k: np.array([step[k] for step in self._episode]) for k in self._episode[0]
        }
        np.savez_compressed(filename, **episode)
    

class EpisodeName:

    def __init__(self, env):
        self._env = env
        self._timestamp = None
        self._unlocked = None
        self._length = None
    
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        self._timestamp = None
        self._unlocked = None
        self._length = 0
        return obs
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._length += 1
        if done:
            self._timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            self._unlocked = sum(int(v >= 1) for v in info['achievements'].values())
        return obs, reward, done, info
    
    @property
    def episode_name(self):
        return f'{self._timestamp}-ach{self._unlocked}-len{self._length}'
