class MediSimConfig(object):
    def __init__(
            self,
            total_vocab_size=59,
            code_vocab_size=56,
            special_vocab_size=3,
      
            cardiopulmonary_vocab_size=20,
            pulmonary_vocab_size=16,
            pleural_vocab_size=8,
            miscellaneous_vocab_size=12,    
            
            n_positions=12,
            n_ctx=12,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            
            batch_size=256,
            batch_size_gen=32,
            plus_batch_size=64,
            sample_batch_size=256,
            epoch=250,
            patience=3,
            lr=1e-4,
            
            epoch_gen=250,
            lr_gen=1e-3,
            image_dim=256,
            n_channels=1,
            embed_dim = 128,
            beta_start = 0.0015,
            beta_end = 0.02,
            num_timesteps = 1000,
            image_dir = '/srv/local/data/MIMIC-CXR/images',
    ):
        self.total_vocab_size = total_vocab_size
        self.code_vocab_size = code_vocab_size
        self.special_vocab_size = special_vocab_size
        self.cardiopulmonary_vocab_size = cardiopulmonary_vocab_size
        self.pulmonary_vocab_size = pulmonary_vocab_size
        self.pleural_vocab_size = pleural_vocab_size
        self.miscellaneous_vocab_size = miscellaneous_vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.batch_size = batch_size
        self.batch_size_gen = batch_size_gen
        self.plus_batch_size = plus_batch_size
        self.sample_batch_size = sample_batch_size
        self.epoch = epoch
        self.lr = lr
        self.patience = patience
        self.lr_gen = lr_gen
        self.epoch_gen = epoch_gen
        self.image_dim = image_dim
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps
        self.image_dir = image_dir