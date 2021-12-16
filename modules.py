import torch.nn as nn
import torch
# from model import AdditiveAttentionEdgeDecoder, AttentionUpsampler

def format_linear(tensor):
    return tensor.permute(0, 2, 3, 1)

def deformat_linear(tensor):
    return tensor.permute(0, 3, 1, 2)

class DecoderAttentionSingle(nn.Module):
    def __init__(self, enc_channels, dec_channels):
        super(DecoderAttentionSingle, self).__init__()
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.W_encoder = nn.Linear(enc_channels, enc_channels, bias = True)
        self.W_decoder = nn.Linear(dec_channels, enc_channels, bias = True)
        self.agg_encoder = nn.Linear(enc_channels, 1, bias = True)
        self.attention_layer = nn.Linear(2 * enc_channels, enc_channels, bias= True)
        self.values_conv = nn.Conv2d(dec_channels, enc_channels, 3, 1, 1, bias = True)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace = True)

    def get_eight_neighbourhood(self, i, j, queries, keys, enc_features):
        batch_size, channels, h, w = keys.shape
        query = queries[:, :, i, j]
        attn_weights, enc_vectors = [], []
        for r in range(-1, 2):
            for c in range(-1, 2):
                if i+r >= 0 and j+c >= 0 and i+r < h and j+c < w:
                    key = keys[:, :, i + r, j + c]
                    enc_vector = enc_features[:, :, i + r, j + c]
                    attn_weight = self.agg_encoder(torch.tanh(query + key))
                    attn_weights.append(attn_weight); enc_vectors.append(enc_vector)
        # Obtain attn weights and encoded vector as tensors.
        attn_weights_tensor, enc_vectors_tensor = torch.stack(attn_weights, dim = -1), \
                                                  torch.stack(enc_vectors, dim = -1)
        # do softmax weighing of weights
        softmax_weighted_attn = nn.functional.softmax(attn_weights_tensor, dim = -1)
        # combine weights
        return torch.sum(torch.mul(softmax_weighted_attn, enc_vectors_tensor), dim = -1, keepdim = False)

    def forward(self, encoder_features, decoder_features):
        # encoder features will be of shape b x 16 x 16 x 1024
        # decoder features will be of shape b x 16 x 16 x 512
        decoder_linear, encoder_linear = format_linear(decoder_features), format_linear(encoder_features)
        queries = deformat_linear(self.W_decoder(decoder_linear))
        keys = deformat_linear(self.W_encoder(encoder_linear))
        values = self.values_conv(decoder_features)
        attended = []
        batch_size, _ , spatial_h, spatial_w  = queries.shape
        for i in range(spatial_h):
            for j in range(spatial_w):
                attn_vector = self.get_eight_neighbourhood(i, j, queries, keys, encoder_features)
                attended.append(torch.cat([values[:, :, i, j], attn_vector], dim = -1))
        attended_tensor = torch.stack(attended, dim=-1).reshape(batch_size, (2 * self.enc_channels), spatial_h, spatial_w)
        attended_tensor_linear = format_linear(attended_tensor)
        return self.leaky_relu(deformat_linear(self.attention_layer(attended_tensor_linear)))

class UpsamplerAttentionSingle(nn.Module):
    def __init__(self, context_channels, feat_channels):
        super(UpsamplerAttentionSingle, self).__init__()
        self.W_context = nn.Linear(context_channels, feat_channels)
        self.W_feat = nn.Linear(feat_channels, feat_channels)
        self.values_conv = nn.Conv2d(feat_channels, context_channels, 3, 1, 1, bias = True)

    def get_upsampled_features(self, keys_local, query, value, context_native, mainstream_native):
        batch_size = keys_local.shape[0]
        attended = []
        for i in range(2):
            for j in range(2):
                key = keys_local[:, :, i, j]
                # will result in batch_size x 1 vector for context attention
                contextual_attn_weight = torch.bmm(mainstream_native.reshape(batch_size, 1, -1), key.reshape(batch_size, -1, 1))  # makes context to feat size - 32, 1024
                # will give us batch_size x 1 vector for self attention
                self_attention_weight = torch.bmm(mainstream_native.reshape(batch_size, 1, -1), query.reshape(batch_size, -1, 1))
                # attention weights together
                softmax_attn_weights = nn.functional.softmax(torch.stack([contextual_attn_weight, self_attention_weight], dim=-1), dim=-1) # batch_size, 1, 2 weights
                # combine this with actual key-values.
                attention_vector = torch.sum(torch.mul([softmax_attn_weights, torch.stack([context_native, value], dim=-1)]), dim=-1, keepdim = False)
                attended.append(attention_vector)
        return torch.stack(attended, dim=-1)

    def forward(self, contexts, mainstream):
            # 1. take (x,y) from here. example (0, 1)
            #     a. find corresponding 4 coordinates from the other map.
            # 2. perform dot products between that and this, frame final value.
        # pdb.set_trace()
        batch_size, _, spatial_h, spatial_w = mainstream.size()
        values = self.values_conv(mainstream)
        attention_output = torch.zeros_like(contexts)
        context_linear, mainstream_linear = format_linear(contexts), \
                                    format_linear(mainstream)
        keys, queries = deformat_linear(self.W_context(context_linear)), \
                                      deformat_linear(self.W_feat(mainstream_linear))
        for x in range(spatial_h):
            for y in range(spatial_w):
                attention_output[:, :, 2*x: 2*x+2, 2*y: 2*y+2] = self.get_upsampled_features(keys_local= keys[:, :, 2*x:2*x+2, 2*y:2*y+2],
                                                                                             query = queries[:, :, x, y],
                                                                                             value= values[:, :, x, y],
                                                                                             context_native= context[:, :, 2*x:2*x+2, 2*y:2*y+2],
                                                                                             mainstream_native= mainstream[:, :, x, y]
                                                                                             )
        return attention_output



class UpsamplerAttention(nn.Module):
    def __init__(self, context_channels, feat_channels, num_contexts, target = 'context'):
        super(UpsamplerAttention, self).__init__()
        self.context_channels = context_channels
        self.feat_channels = feat_channels
        self.num_contexts = num_contexts
        self.W_context = nn.Conv2d(context_channels, context_channels, 1, 1, 0, bias = True) #nn.Linear(context_channels, feat_channels)
        self.W_feat = nn.Conv2d(feat_channels, context_channels, 1, 1, 0, bias = True)
        self.W_key_feat = nn.Conv2d(feat_channels, context_channels, 1, 1, 0, bias = True)
        self.values_conv = nn.Conv2d(feat_channels, context_channels, 3, 1, 1, bias = True)

    def get_upsampled_features(self, query, value, mainline_key, contexts_keys, contexts_features):
        # query will be of shape (batch_size , dims)
        attended = []
        for i in range(2):
            for j in range(2):
                keys = [context_keys[:, :, i, j] for context_keys in contexts_keys] + [mainline_key] # each key is [batch_size, 1024]
                native_features = [context_features[:, :, i, j] for context_features in contexts_features] + [value]

                keys_tensor = torch.stack(keys, dim =1)  # (batch, 10 contexts, 1024)
                native_features_tensor = torch.stack(native_features, dim=1) # (batch, 10 contexts, 1024)

                # attn mechanism
                attn_weights = torch.bmm(keys_tensor, query.unsqueeze(dim=-1))  # will be of shape batch_size x 10 x 1
                softmax_attn_weights = nn.functional.softmax(attn_weights, dim=1)
                attn_vector = torch.sum(torch.mul(softmax_attn_weights, native_features_tensor), dim=1, keepdim=False)   # will be batch_size x 1024

                attended.append(attn_vector)
        return torch.stack(attended, dim=-1)

    def forward(self, contexts, mainstream):
        '''
        contexts: is a list of context tensors
        mainstream: is a mainstream tensor.
        '''
        batch_size, _, spatial_h, spatial_w = mainstream.shape
        queries = self.W_feat(mainstream)
        mainstream_keys = self.W_key_feat(mainstream)
        contexts_keys = [self.W_context(context) for context in contexts]
        values = self.values_conv(mainstream)

        attention_output = torch.zeros_like(contexts[0])
        for x in range(spatial_h):
            for y in range(spatial_w):
                upsampled_features = self.get_upsampled_features(query=queries[:, :, x, y],
                                                                 value=values[:, :, x, y],
                                                                 mainline_key=mainstream_keys[:, :, x, y],
                                                                 contexts_keys=[context_keys[:, :, 2*x: 2*x+2, 2*y: 2*y+2] for context_keys in contexts_keys],
                                                                 contexts_features= [context_features[:, :, 2*x: 2*x+2, 2*y: 2*y+2] for context_features in contexts]
                                                                )
                attention_output[:, :, 2*x: 2*x+2, 2*y: 2*y+2] = upsampled_features.view(batch_size, self.context_channels, 2, 2)
        return attention_output


class DecoderAttention(nn.Module):
    def __init__(self, enc_channels, dec_channels, num_contexts):
            super(DecoderAttention, self).__init__()
            self.enc_channels = enc_channels
            self.dec_channels = dec_channels
            self.num_contexts = num_contexts
            self.W_encoder = nn.Conv2d(enc_channels, enc_channels, 1, 1, 0, bias = True)
            self.W_decoder = nn.Conv2d(dec_channels, enc_channels, 1, 1, 0, bias = True)
            self.agg_encoder = nn.Linear(enc_channels, 1, bias = True)
            self.attention_layer = nn.Conv2d((num_contexts + 1) * enc_channels, enc_channels, 1, 1, 0, bias = True)
            self.values_conv = nn.Conv2d(dec_channels, enc_channels, 3, 1, 1, bias = True)
            self.leaky_relu = nn.LeakyReLU(0.2, inplace = True)

    def get_query_attention(self, query, contexts_keys, contexts_features):
        '''
        query: is a single vector in the decoder end
        contexts_keys: is a list of list. outer list is the context. inner list is the spatial neighbourhood points corresponding to query
        contexts_features: gives the base features for every context. is the same as context_keys, but without being applied upon the W_encoder.
        '''
        contexts_attented_vectors = []
        for context, context_features in zip(contexts_keys, contexts_features):
            attn_weights = [ self.agg_encoder(torch.tanh(query + key)) for key in context]
            attn_weights_tensor, contexts_features_tensor = torch.stack(attn_weights, dim=-1), \
                                                            torch.stack(context_features, dim=-1)
            softmax_weighted_attn = nn.functional.softmax(attn_weights_tensor, dim=-1)
            # combine attn weights with context_features
            contexts_attented_vectors.append(torch.sum(torch.mul(softmax_weighted_attn, contexts_features_tensor), dim=-1, keepdim=False))
        return contexts_attented_vectors

    def get_eight_neighbourhood(self, i, j, h, w):
        points = []
        for r in range(-1, 2):
            for c in range(-1, 2):
                if i+r >= 0 and j+c >= 0 and i+r < h and j+c < w:
                    points.append((i+r, j + c))
        return points

    def forward(self, contexts, decoded_features):
        # pdb.set_trace()
        queries = self.W_decoder(decoded_features)
        contexts_keys = [self.W_encoder(context) for context in contexts]
        values = self.values_conv(decoded_features)

        batch_size, _, spatial_h, spatial_w = queries.shape

        attended = []
        for i in range(spatial_h):
            for j in range(spatial_w):
                points = self.get_eight_neighbourhood(i , j, spatial_h, spatial_w)
                query = queries[:, :, i, j]
                contexts_keys_neighbourhood = [ [context_keys[:, :, p1, p2] for (p1, p2) in points] for context_keys in contexts_keys ]
                contexts_features = [ [context[:, :, p1, p2] for (p1, p2) in points] for context in contexts ]
                attention_vectors = self.get_query_attention(query, contexts_keys_neighbourhood, contexts_features)
                attended.append(torch.cat([ values[:, :, i, j] ] + attention_vectors , dim = -1))
        attended_tensor = torch.stack(attended, dim=-1).reshape(batch_size, (self.num_contexts+1) * self.enc_channels, spatial_h, spatial_w)
        return self.leaky_relu(self.attention_layer(attended_tensor))


'''
class AttentionUpsamplerEdgeDecoderNative(AdditiveAttentionEdgeDecoder, AttentionUpsampler):
    def __init__(self):
        super().__init__()
        self.attn_upsample3 = UpsamplerAttentionSingle(128, 256)
        self.attn_upsample4 = UpsamplerAttentionSingle(64, 128)
        self.attn_upsample5 = UpsamplerAttentionSingle(32, 64)

        self.edge_decoder1 = DecoderAttentionSingle(enc_channels=128, dec_channels=128)
        self.edge_decoder2 = DecoderAttentionSingle(enc_channels=64, dec_channels=64)
        self.edge_decoder3 = DecoderAttentionSingle(enc_channels=32, dec_channels=32)

        self.classifier_conv = nn.Conv2d(32, 2, 3, 1, 1, bias = True)

        self.edge_prelim_conv = nn.Conv2d(1, 32, 3, 1, 1, bias=False)  # 32
        self.edge1_conv = nn.Conv2d(32, 64, 3, 2, 1, bias=False)       # 64


    def forward(self, input):
        input, edge_map = input
        edge_prelim = self.conv_layer(edge_map, self.edge_prelim_conv, self.batch_norm32, self.leaky_relu)  # 256 x 256 x 32
        edge1 = self.conv_layer(edge_prelim, self.edge1_conv, self.batch_norm64, self.leaky_relu)  #  128 x 128 x 64
        edge2 = self.conv_layer(edge1, self.edge2_conv, self.batch_norm128, self.leaky_relu)       # 64 x 64 x 128

        prelim = self.conv_layer(input, self.encode0, self.batch_norm32, self.leaky_relu)   # 256 x 256 x 32
        x1 = self.conv_layer(prelim, self.encode1, self.batch_norm64, self.leaky_relu)      # 128 x 128 x 64
        x2 = self.conv_layer(x1, self.encode2, self.batch_norm128, self.leaky_relu)         # 64 x 64 x 128
        x3 = self.conv_layer(x2, self.encode3, self.batch_norm256, self.leaky_relu)         # 32 x 32 x 256
        x4 = self.conv_layer(x3, self.encode4, self.batch_norm512, self.leaky_relu)         # 16 x 16 x 512
        encoded = self.conv_layer(x4, self.encode5, self.batch_norm1024, self.leaky_relu)   # 8 x 8 x 1024

        z1 = self.attn_upsample1(contexts=x4, mainstream=encoded) # of shape 16 x 16 x 512
        z2 = self.attn_upsample2(contexts=x3, mainstream=z1)   # is of shape 32 x 32 x 256
        z3 = self.attn_upsample3(contexts=x2, mainstream=z2)  # is of shape 64 x 64 x 128
        # single step decoder attention
        decoded_features1 = self.edge_decoder1(edge2, z3)   # 64 x 64 x 128
        # upsample that
        z4 = self.attn_upsample4(contexts=x1, mainstream = decoded_features1)   # 128 x 128 x 64
        # single step decoder attention
        # pdb.set_trace()
        decoded_features2 = self.edge_decoder2(edge1, z4)  # 128 x 128 x 64
        # upsample to 256
        z5 = self.attn_upsample5(prelim, mainstream=decoded_features2)  # 256 x 256 x 32
        # decode at 256 scale
        decoded_features3 = self.edge_decoder3(edge_prelim, z5) # 256 x 256 x 32

        # classifier layer
        z6 = self.conv_layer(decoded_features3, self.classifier_conv, self.batch_norm2, self.leaky_relu, upsample=False)
        return z6
'''


class UpsamplerAttentionParallel(nn.Module):
    def __init__(self, context_channels, feat_channels, num_contexts, target = 'context'):
        super(UpsamplerAttentionParallel, self).__init__()
        self.context_channels = context_channels
        self.feat_channels = feat_channels
        self.num_contexts = num_contexts
        self.W_context = nn.Conv2d(context_channels, context_channels, 1, 1, 0, bias = True) #nn.Linear(context_channels, feat_channels)
        self.W_feat = nn.Conv2d(feat_channels, context_channels, 1, 1, 0, bias = True)
        self.W_key_feat = nn.Conv2d(feat_channels, context_channels, 1, 1, 0, bias = True)
        self.values_conv = nn.Conv2d(feat_channels, context_channels, 3, 1, 1, bias = True)
        self.upsample = nn.Upsample(scale_factor=2, mode = 'nearest')

    def forward(self, contexts, mainstream):
        '''
        contexts: is a list of context tensors
        mainstream: is a mainstream tensor.
        '''
        batch_size, _, spatial_h, spatial_w = contexts[0].shape
        queries = self.W_feat(mainstream)

        mainstream_keys = self.W_key_feat(mainstream)
        mainstream_keys_nearest_upsample = self.upsample(mainstream_keys)

        contexts_keys = self.W_context(torch.stack(contexts + [mainstream_keys_nearest_upsample], dim=1) \
                            .view(batch_size * (self.num_contexts+1), self.context_channels, spatial_h, spatial_w)) \
                            .view(batch_size, self.num_contexts + 1, self.context_channels, spatial_h, spatial_w) \
                            .permute(0, 3, 4, 1, 2)
                    # assume each context of shape B x 16 x 16 x 4 x 512
        values = self.values_conv(mainstream)

        queries_nearest_upsample = self.upsample(queries).unsqueeze(dim=-1).permute(0, 2, 3, 1, 4)   # assume each context of shape B x 16 x 16 x 512 x 1
        values_nearest_upsample = self.upsample(values)

        attn_weights = torch.matmul(contexts_keys, queries_nearest_upsample).transpose(dim0=3, dim1=4)     # B x 16 x 16 x 4 x 1 transposed to b x 16 x 16 x 1 x 4
        softmax_attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attention_output = torch.sum(torch.mul(torch.stack(contexts + [values_nearest_upsample], dim=-1).permute(0, 2, 3, 1, 4), softmax_attn_weights), dim=-1, keepdim=False)  # b x 16 x 16 x 512

        # get the channel axis to 1.
        return attention_output.permute(0, 3, 1, 2)



if __name__ == '__main__':
    model = AttentionUpsamplerEdgeDecoderNative()
    x = torch.randn(10, 1, 256, 256)
    edges = torch.randn(10, 1, 256, 256)
    out = model((x, edges))
    print(out.shape)