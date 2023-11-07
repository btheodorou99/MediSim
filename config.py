class MediSimConfig(object):
    def __init__(
            self,
            total_vocab_size=13222,
            code_vocab_size=13219,
            special_vocab_size=3,
      
            diagnosis_vocab_size=6984,
            procedure_vocab_size=2032,
            medication_vocab_size=4203,
            
            phenotype_labels=25,
            
            n_positions=44,
            n_ctx=44,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            
            batch_size=48,
            sample_batch_size=256,
            epoch=50,
            lr=1e-4,
    ):
        self.total_vocab_size = total_vocab_size
        self.code_vocab_size = code_vocab_size
        self.special_vocab_size = special_vocab_size
        self.diagnosis_vocab_size = diagnosis_vocab_size
        self.procedure_vocab_size = procedure_vocab_size
        self.medication_vocab_size = medication_vocab_size
        self.phenotype_labels = phenotype_labels
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