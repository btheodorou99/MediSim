class MediSimConfig(object):
    def __init__(
            self,
            total_vocab_size=13128,
            code_vocab_size=13125,
            special_vocab_size=3,
      
            diagnosis_vocab_size=6918,
            procedure_vocab_size=2011,
            medication_vocab_size=4196,
            
            n_positions=8,
            n_ctx=8,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            
            batch_size=18,
            sample_batch_size=256,
            epoch=50,
            patience=3,
            lr=1e-4,
            
            epoch_gen=250,
            n_gen_positions=512,
            batch_size_gen=64,
            discriminator_batch_size=64,
            word_vocab_size=28125, # Min 101, Max 28124, +1 for [END] Token
            n_embed_positions=512,
            n_embed_layer=4,
    ):
        self.total_vocab_size = total_vocab_size
        self.code_vocab_size = code_vocab_size
        self.special_vocab_size = special_vocab_size
        self.diagnosis_vocab_size = diagnosis_vocab_size
        self.procedure_vocab_size = procedure_vocab_size
        self.medication_vocab_size = medication_vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.batch_size = batch_size
        self.sample_batch_size = sample_batch_size
        self.epoch = epoch
        self.lr = lr
        self.patience = patience
        self.epoch_gen = epoch_gen
        self.n_gen_positions = n_gen_positions
        self.batch_size_gen = batch_size_gen
        self.discriminator_batch_size = discriminator_batch_size
        self.word_vocab_size = word_vocab_size
        self.n_embed_positions = n_embed_positions
        self.n_embed_layer = n_embed_layer