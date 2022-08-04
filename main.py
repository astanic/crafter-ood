import framework
import tasks


def register_args(parser: framework.helpers.ArgumentParser):

    # General
    parser.add_argument('-task', default='crf_ppo')
    parser.add_argument('-profile_name', type=str, help='Train dir name')
    parser.add_argument('-keep_alive', default=False, help='After training end, keeps process alive (useful for looking at the TB logs)')
    parser.add_argument('-seed', default='none', parser=parser.int_or_none_parser)
    parser.add_argument('-gpu', default='auto', help='use this gpu')
    parser.add_argument('-wandb_project_name', default='crafter-ood')
    parser.add_argument('-logger.type', default='wandb')
    parser.add_argument('-logger.sb3.sync_tb', default=True)
    parser.add_argument('-logger.sb3.monitor_gym', default=True)
    parser.add_argument('-logger.sb3.save_code', default=True)

    # Training
    parser.add_argument('-max_train_steps', default=1_000_000)
    parser.add_argument('-log_every_n_episodes', default=100)
    parser.add_argument('-opt_all.lr', default=1e-4)

    # Environment
    parser.add_argument('-crf.size', default=64)
    parser.add_argument('-crf.max_ep_len', default=10_000)
    parser.add_argument('-crf.render_scoreboard', default=True)
    parser.add_argument('-crf.save_video', default=False)
    parser.add_argument('-eval_n_steps', default=50_000)
    parser.add_argument('-eval_n_episodes', default=5)
    parser.add_argument('-el_vars', default='')
    parser.add_argument('-el_freq_train', default='100,0,0,0')
    parser.add_argument('-el_freq_valid', default='100,0,0,0')
    parser.add_argument('-el_app_freq_train', default='sssss')
    parser.add_argument('-el_app_freq_valid', default='sssss')

    # Model
    parser.add_argument('-rsn.type', default='', choice=['', 'oc_sa', 'oc_ca'])
    parser.add_argument('-oc_sa.d_model', default=256)
    parser.add_argument('-oc_sa.n_head', default=8)
    parser.add_argument('-oc_sa.d_head', default=32)
    parser.add_argument('-oc_sa.d_inner', default=128)
    parser.add_argument('-oc_sa.dropout', default=0.0)
    parser.add_argument('-oc_sa.dropatt', default=0.1)
    parser.add_argument("-oc_ca.n_slot", default=8)
    parser.add_argument("-oc_ca.d_slot", default=256)

    # stable-baselines3 arguments
    parser.add_argument('-ppo.recurrent', default=False)
    parser.add_argument('-ppo.gpu', default=0)
    parser.add_argument('-ppo.n_steps', default=4096)
    parser.add_argument('-ppo.batch_size', default=128)
    parser.add_argument('-ppo.n_epochs', default=4)
    parser.add_argument('-ppo.gamma', default=0.95)
    parser.add_argument('-ppo.gae_lambda', default=0.65)
    parser.add_argument('-ppo.clip_range', default=0.2)
    parser.add_argument('-ppo.clip_range_vf', default=None, type=float)
    parser.add_argument('-ppo.normalize_advantage', default=True)
    parser.add_argument('-ppo.ent_coef', default=0.0)
    parser.add_argument('-ppo.vf_coef', default=0.5)
    parser.add_argument('-ppo.max_grad_norm', default=0.5)
    parser.add_argument('-ppo.target_kl', default=None, type=float)
    parser.add_argument('-fe.type', default='i2s')
    parser.add_argument('-fe.patch_size', default=8)
    parser.add_argument('-fe.patch_stride', default=8)
    parser.add_argument('-fe.precnn.type', default='none')
    parser.add_argument('-fe.precnn.d_out', default=64)
    parser.add_argument('-fe.cnnmap.type', default='cnn')
    parser.add_argument('-fe.cnnmap.d_out', default=64)
    parser.add_argument('-fe.outmap.type', default='linear')
    parser.add_argument('-fe.outmap.d_out', default=512)

    # Tasks
    parser.add_profile([
        parser.Profile('ppo_cnn', {
            'task': 'crf_ppo',
        }),

        parser.Profile('lstm_cnn', {
            'ppo.recurrent': True,
        }, include='ppo_cnn'),
 
        parser.Profile('ppo_spcnn', {
            'fe.cnnmap.type': 'cnn-sa',
            'fe.cnnmap.d_out': 64,  # 64, 16, 4
            'fe.outmap.type': 'linear',
            'fe.outmap.d_out': 512,
        }, include='ppo_cnn'),

        parser.Profile('lstm_spcnn', {
            'ppo.recurrent': True,
        }, include='ppo_spcnn'),

        parser.Profile('oc_sa', {
            'crf.size': 72,
            'rsn.type': 'oc_sa',
            'fe.patch_size': 12,
            'fe.patch_stride': 8,
            'fe.precnn.type': 'cnn-sa',
            'fe.cnnmap.type': 'patch-none',
            'fe.cnnmap.d_out': 64,
            'fe.outmap.type': 'none',
            'fe.outmap.d_out': 64,
        }, include='ppo_cnn'),

        parser.Profile('oc_ca', {
            'rsn.type': 'oc_ca',
            'fe.patch_size': 16,
            'fe.patch_stride': 16,
        }, include='oc_sa'),
    ])


def main():
    helper = framework.helpers.TrainingHelper(register_args=register_args)
    task = tasks.RLTaskCRF(helper)
    task.train()
    task.test()
    helper.finish()

if __name__ == '__main__':
    main()
