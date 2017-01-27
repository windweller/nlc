from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import json
import string
import itertools

import numpy as np
from six.moves import xrange
import tensorflow as tf
import tflearn

import nlc_model
import nlc_data
import util

from nlc_data import PAD_ID
from util import padded, add_sos_eos, get_tokenizer, pair_iter

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.0003, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 40, "Number of epochs to pre-train actor model.")
tf.app.flags.DEFINE_integer("critic_epochs", 1, "Number of epochs to pre-train critic.")
tf.app.flags.DEFINE_integer("rl_epochs", 1, "Number of epochs to train entire rl algorithm.")
tf.app.flags.DEFINE_integer("size", 400, "Size of each model layer.")  # 400
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("max_vocab_size", 40000, "Vocabulary size limit.")
tf.app.flags.DEFINE_integer("delay_eval", 15, "pre-train how many epochs to start evaluating")
# tf.app.flags.DEFINE_integer("max_seq_len", 200, "Maximum sequence length.")
tf.app.flags.DEFINE_integer("max_seq_len", 32, "Maximum sequence length.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("tokenizer", "CHAR", "BPE / CHAR / WORD.")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_string("evaluate", "CER", "BLEU / CER")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_bool("rl_only", False, "flag True to only train rl portion")
tf.app.flags.DEFINE_bool("sup_only", False, "flag True to only train supervised portion")

tf.app.flags.DEFINE_integer("beam_size", 8, "Size of beam.")
tf.app.flags.DEFINE_string("lmfile", None, "arpa file of the language model.")
tf.app.flags.DEFINE_float("alpha", 0.3, "Language model relative weight.")

FLAGS = tf.app.flags.FLAGS
vocab, rev_vocab = None, None
lm = None


def create_model(vocab_size, forward_only, model_name):
    model = nlc_model.NLCModel(
        vocab_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.dropout, FLAGS,
        forward_only=forward_only, optimizer=FLAGS.optimizer)
    logging.info("Creating model %s" % model_name)
    return model


def initialize_models(session, model):
    # call this to initialize all models, only need to pass in one model
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))

def restore_models(session, model):
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        return True
    else:
        return False

def validate(model, sess, x_dev, y_dev):
    valid_costs, valid_lengths = [], []
    for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_dev, y_dev, FLAGS.batch_size,
                                                                            FLAGS.num_layers):
        cost = model.test(sess, source_tokens, source_mask, target_tokens, target_mask)
        valid_costs.append(cost * target_mask.shape[1])
        valid_lengths.append(np.sum(target_mask[1:, :]))
    valid_cost = sum(valid_costs) / float(sum(valid_lengths))
    return valid_cost


def lmscore(ray, v):
    if lm is None:
        return 0.0

    sent = ' '.join(ray[3])
    if len(sent) == 0:
        return 0.0

    if v == nlc_data.EOS_ID:
        return sum(w[0] for w in list(lm.full_scores(sent, eos=True))[-2:])
    elif rev_vocab[v] in string.whitespace:
        return list(lm.full_scores(sent, eos=False))[-1][0]
    else:
        return 0.0


def zip_input(beam):
    inp = np.array([ray[2][-1] for ray in beam], dtype=np.int32).reshape([1, -1])
    return inp


def zip_state(beam):
    if len(beam) == 1:
        return None  # Init state
    return [np.array([(ray[1])[i, :] for ray in beam]) for i in xrange(FLAGS.num_layers)]


def unzip_state(state):
    beam_size = state[0].shape[0]
    return [np.array([s[i, :] for s in state]) for i in xrange(beam_size)]


def tokenize(sent, vocab, depth=FLAGS.num_layers):
  align = pow(2, depth - 1)
  token_ids = nlc_data.sentence_to_token_ids(sent, vocab, get_tokenizer(FLAGS))
  ones = [1] * len(token_ids)
  pad = (align - len(token_ids)) % align

  token_ids += [nlc_data.PAD_ID] * pad
  ones += [0] * pad

  source = np.array(token_ids).reshape([-1, 1])
  mask = np.array(ones).reshape([-1, 1])

  return source, mask

# CPU
def beam_step(beam, candidates, decoder_output, zipped_state, max_beam_size):
    logprobs = (decoder_output).squeeze(axis=0)  # [batch_size x vocab_size]
    newbeam = []

    for (b, ray) in enumerate(beam):
        prob, _, seq, low = ray
        for v in reversed(list(np.argsort(logprobs[b, :]))):  # Try to look at high probabilities in each ray first

            newprob = prob + logprobs[b, v] + FLAGS.alpha * lmscore(ray, v)

            if rev_vocab[v] in string.whitespace:
                newray = (newprob, zipped_state[b], seq + [v], low + [''])
            elif v >= len(nlc_data._START_VOCAB):
                newray = (newprob, zipped_state[b], seq + [v], low[:-1] + [low[-1] + rev_vocab[v]])
            else:
                newray = (newprob, zipped_state[b], seq + [v], low)

            if len(newbeam) > max_beam_size and newprob < newbeam[0][0]:
                continue

            if v == nlc_data.EOS_ID:
                candidates += [newray]
                candidates.sort(key=lambda r: r[0])
                candidates = candidates[-max_beam_size:]
            else:
                newbeam += [newray]
                newbeam.sort(key=lambda r: r[0])
                newbeam = newbeam[-max_beam_size:]


def decode_beam_cpu(model, sess, encoder_output, max_beam_size):
    state, output = None, None
    beam = [(0.0, None, [nlc_data.SOS_ID],
             [''])]  # (cumulative log prob, decoder state, [tokens seq], ['list', 'of', 'words'])

    candidates = []
    while True:
        output, attn_map, state = model.decode(sess, encoder_output, zip_input(beam), decoder_states=zip_state(beam))
        beam, candidates = beam_step(beam, candidates, output, unzip_state(state), max_beam_size)
        if beam[-1][0] < 1.5 * candidates[0][0]:
            # Best ray is worse than worst completed candidate. candidates[] cannot change after this.
            break
    # print_beam(candidates, 'Candidates')
    finalray = candidates[-1]
    return finalray[2]


def cer_evaluate(model, sess, x_dev, y_dev, curr_epoch, sample_rate=0.005, delay_sampling=10):
    valid_cers = []
    for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_dev, y_dev, 1,
                                                                            FLAGS.num_layers):
        # Encode
        encoder_output = model.encode(sess, source_tokens, source_mask)
        # Decode
        # beam decode might only work on GPU...so we use greedy decode
        beam_toks, probs = decode_beam(model, sess, encoder_output, 1)
        # De-tokenize
        beam_strs = detokenize(beam_toks, rev_vocab)
        target_str = detokenize_tgt(target_tokens, rev_vocab)
        # Language Model ranking
        best_str = lm_rank(beam_strs, probs)  # return first MML-based string

        valid_cers.append(compute_cer(target_str, best_str))
        if curr_epoch >= delay_sampling:
            if np.random.sample() <= sample_rate:  # don't know performance penalty of np.random.sample()
                print("sampled target str: %s" % target_str)
                print("sampled best str: %s" % best_str)

    mean_valid_cer = sum(valid_cers) / float(len(valid_cers))
    return mean_valid_cer


def build_data(fnamex, fnamey, num_layers, max_seq_len):
    fdx, fdy = open(fnamex), open(fnamey)
    x_token_list = []
    y_token_list = []

    # we need to fill in the entire dataset
    linex, liney = fdx.readline(), fdy.readline()

    while linex and liney:
        x_tokens, y_tokens = util.tokenize(linex), util.tokenize(liney)

        # this is not truncating...just ignoring
        if len(x_tokens) < max_seq_len and len(y_tokens) < max_seq_len:
            x_token_list.append(x_tokens)
            y_token_list.append(y_tokens)

        linex, liney = fdx.readline(), fdy.readline()

    y_token_list = add_sos_eos(y_token_list)  # shift y by 1 position
    x_padded, y_padded = padded(x_token_list, num_layers), padded(y_token_list, 1)

    source_tokens = np.array(x_padded).T
    source_mask = (source_tokens != PAD_ID).astype(np.int32)
    target_tokens = np.array(y_padded).T
    target_mask = (target_tokens != PAD_ID).astype(np.int32)

    return source_tokens, source_mask, target_tokens, target_mask


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_cer(a, b):
    # a must be ground truth
    ground_truth_len = float(len(a))
    return levenshtein(a, b) / ground_truth_len


def detokenize(sents, reverse_vocab):
    # TODO: char vs word
    def detok_sent(sent):
        outsent = ''
        for t in sent:
            if t >= len(nlc_data._START_VOCAB):
                outsent += reverse_vocab[t]
        return outsent

    return [detok_sent(s) for s in sents]


def detokenize_tgt(toks, reverse_vocab):
    outsent = ''
    for i in range(toks.shape[0]):
        if toks[i] >= len(nlc_data._START_VOCAB) and toks[i] != nlc_data._PAD:
            outsent += reverse_vocab[toks[i][0]]
    return outsent


def lm_rank(strs, probs):
    if lm is None:
        return strs[0]
    a = FLAGS.alpha
    lmscores = [lm.score(s) / (1 + len(s.split())) for s in strs]
    probs = [p / (len(s) + 1) for (s, p) in zip(strs, probs)]
    for (s, p, l) in zip(strs, probs, lmscores):
        print(s, p, l)

    rescores = [(1 - a) * p + a * l for (l, p) in zip(lmscores, probs)]
    rerank = [rs[0] for rs in sorted(enumerate(rescores), key=lambda x: x[1])]
    generated = strs[rerank[-1]]
    lm_score = lmscores[rerank[-1]]
    nw_score = probs[rerank[-1]]
    score = rescores[rerank[-1]]
    return generated  # , score, nw_score, lm_score


def decode_beam(model, sess, encoder_output, max_beam_size):
    toks, probs = model.decode_beam(sess, encoder_output, beam_size=max_beam_size)
    return toks.tolist(), probs.tolist()


def train_seq2seq(model, sess, x_dev, y_dev, x_train, y_train):
    print('Initial validation cost: %f' % validate(model, sess, x_dev, y_dev))

    if False:
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        print("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

    epoch = 0
    previous_losses = []
    exp_cost = None
    exp_length = None
    exp_norm = None
    while (FLAGS.epochs == 0 or epoch < FLAGS.epochs):
        epoch += 1
        current_step = 0

        ## Train
        for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_train, y_train, FLAGS.batch_size,
                                                                                FLAGS.num_layers):
            # Get a batch and make a step.
            tic = time.time()

            grad_norm, cost, param_norm = model.train(sess, source_tokens, source_mask, target_tokens, target_mask)

            toc = time.time()
            iter_time = toc - tic
            current_step += 1

            lengths = np.sum(target_mask, axis=0)
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)

            if not exp_cost:
                exp_cost = cost
                exp_length = mean_length
                exp_norm = grad_norm
            else:
                exp_cost = 0.99 * exp_cost + 0.01 * cost
                exp_length = 0.99 * exp_length + 0.01 * mean_length
                exp_norm = 0.99 * exp_norm + 0.01 * grad_norm

            cost = cost / mean_length

            if current_step % FLAGS.print_every == 0:
                print(
                    'epoch %d, iter %d, cost %f, exp_cost %f, grad norm %f, param norm %f, batch time %f, length mean/std %f/%f' %
                    (epoch, current_step, cost, exp_cost / exp_length, grad_norm, param_norm, iter_time,
                     mean_length,
                     std_length))

        ## Checkpoint
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

        ## Validate
        valid_cost = validate(model, sess, x_dev, y_dev)

        print("Epoch %d Validation cost: %f" % (epoch, valid_cost))

        ## Evaluate
        if FLAGS.evaluate == "CER":
            # CER evaluate does not do beam-decode with n-gram LM, Max Likelihood decode
            # because we don't have a language model (chop-off is clean-cut)

            # we evaluate on validation set
            cer = cer_evaluate(model, sess, x_dev, y_dev, epoch, delay_sampling=10)
            print("Epoch %d CER: %f" % (epoch, cer))

        if len(previous_losses) > 2 and valid_cost > max(previous_losses[-3:]):
            sess.run(model.learning_rate_decay_op)
        previous_losses.append(valid_cost)
        sys.stdout.flush()

    return model


def decode_greedy(model, sess, encoder_output):
  decoder_state = None
  decoder_input = np.array([nlc_data.SOS_ID, ], dtype=np.int32).reshape([1, 1])

  output_sent = []
  while True:
    decoder_output, _, decoder_state = model.decode(sess, encoder_output, decoder_input, decoder_states=decoder_state)
    token_highest_prob = np.argmax(decoder_output.flatten())
    if token_highest_prob == nlc_data.EOS_ID or len(output_sent) > FLAGS.max_seq_len:
      break
    output_sent += [token_highest_prob]
    decoder_input = np.array([token_highest_prob, ], dtype=np.int32).reshape([1, 1])

  return output_sent


# now batched, similar to rllab's style
# TODO: might want to test this...(right now it seems to work!!!)
def decode_greedy_batch(model, sess, encoder_output, batch_size):
    decoder_state = None
    decoder_input = np.array([nlc_data.SOS_ID, ] * batch_size, dtype=np.int32).reshape([1, batch_size])

    attention = []
    output_sent = np.array([nlc_data.PAD_ID,] * FLAGS.max_seq_len
                           * batch_size, dtype=np.int32).reshape([FLAGS.max_seq_len, batch_size])
    dones = np.array([True,] * FLAGS.batch_size, dtype=np.bool)
    i = 0
    while True:
        decoder_output, attn_map, decoder_state = model.decode(sess, encoder_output, decoder_input,
                                                               decoder_states=decoder_state)
        attention.append(attn_map)
        # decoder_output shape: (1, batch_size, vocab_size)
        token_highest_prob = np.argmax(np.squeeze(decoder_output), axis=1)

        # token_highest_prob shape: (batch_size,)
        mask = token_highest_prob == nlc_data.EOS_ID
        update_dones_indices = np.nonzero(mask)
        # update on newly finished sentence, add EOS_ID
        new_finished = update_dones_indices != dones
        output_sent[i, new_finished] = nlc_data.EOS_ID

        dones[update_dones_indices] = False
        if i >= FLAGS.max_seq_len - 1 or np.sum(np.nonzero(dones)) == 0:
            break

        output_sent[i, dones] = token_highest_prob
        decoder_input = token_highest_prob.reshape([1, batch_size])
        i += 1

    return output_sent


# TODO: test this with newly trained model!
def decompose_reward(a, b):
    # a is ground truth, both are tokenized with padding
    # return shape: (time,)
    reward_hist = np.zeros((FLAGS.max_seq_len), dtype=np.float32)
    reward_gain = np.zeros((FLAGS.max_seq_len), dtype=np.float32)
    if b.size == 0:
        return reward_gain
    for i in range(1, FLAGS.max_seq_len):
        reward = 1 - compute_cer(np.array_str(a[:i]), np.array_str(b[:i]))
        reward_hist[i] = 0 if reward <= 0 else reward
    reward_gain[1:] = np.diff(reward_hist)  # first reward is always 0
    return reward_gain


def clip_after_eos(a, no_eos=False):
    # take in a 1-dim np array, this pads as well
    # mask_nonzero + 1 means we want to keep <EOS>
    # mask_nonzero means we don't want <EOS>
    # no_eos: we don't add eos, this option is for decode() function, not for beam_decode()
    mask = a == nlc_data.EOS_ID
    mask_nonzero = mask.nonzero()[0].tolist()  # only want the first EOS
    if len(mask_nonzero) != 0:  # sometimes it didn't generate an EOS...
        pos = mask_nonzero[0] if no_eos else mask_nonzero[0] + 1
        a[pos:a.shape[0]] = nlc_data.PAD_ID  # pad it
    return a


def process_samples(sess, actor, x, y):
    # this batch things together based on the batch size
    # in the end, we can just izip the arrays, and iterate on them
    rewards, actions_dist, actions, actions_mask = [], [], [], []
    source_tokenss, target_tokenss = [], []
    # actions: (time, batch_size, vocab) # condition on ground truth targets

    # for universal padding, we can iterate through the dataset, and determine the
    # optimal batch_max_len for each batch, then pass in
    # batch_pads can be a list, we keep track of an iterator, and each turn just pass it in

    # not adding sos eos to target_tokens
    # Note: action_dist is [T, batch_size, vocab_size]
    for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x, y, 1,
                                                                            FLAGS.num_layers,
                                                                            add_sos_eos_bool=False):

        source_tokenss.append(np.squeeze(source_tokens).tolist())
        target_tokenss.append(np.squeeze(target_tokens).tolist())

        encoder_output = actor.encode(sess, source_tokens, source_mask)
        best_tok, _ = decode_beam(actor, sess, encoder_output, 1)
        best_tok[0][-1] = nlc_data.EOS_ID  # last data mark as EOS
        padded_best_tok = padded(best_tok, depth=1, batch_pad=32)  # TODO: remember to switch to a univeral pad list

        # way to solve batch problem - pad best_tok!

        decoder_output, _, _ = actor.decode(sess, encoder_output, np.matrix(padded_best_tok).T)

        # decoder_output = np.squeeze(decoder_output)

        tok_highest_prob = np.argmax(np.squeeze(decoder_output), axis=1)
        # clipped_tok_highest_prob = clip_after_eos(tok_highest_prob)  # hmmm, not sure if we should clip after eos
        clipped_tok_highest_prob = tok_highest_prob

        # print("token with highest prob: ")
        # print(clipped_tok_highest_prob)
        # print("target toks: ")
        # print(np.squeeze(target_tokens))

        # TODO: test reward :(
        reward = decompose_reward(np.squeeze(target_tokens), clipped_tok_highest_prob)
        # print(reward)
        rewards.append(reward)

        # need to pad actions and make masks...
        # print("action shape: %s" % (best_tok.shape,))
        # print(best_tok[0])
        # print("action dist shape: %s" % (tok_prob.shape,))

        # print("token len: {}".format(clipped_tok_highest_prob.shape))
        # print("target len: {}".format(target_tokens.shape))
        # print("action dist shape: {}".format(decoder_output.shape))

        actions.append(clipped_tok_highest_prob)
        actions_dist.append(decoder_output)

        if len(rewards) % FLAGS.batch_size == 0:
            # padding problem solved!!
            batch = (np.array(rewards), np.concatenate(actions_dist, axis=1), np.array(actions))

            # notice the transpose for source, not for target
            # notice no sos_eos for target!
            x_padded = np.array(padded(source_tokenss, FLAGS.num_layers)).T
            source_masks = (x_padded != nlc_data.PAD_ID).astype(np.int32)
            y_padded = np.array(padded(target_tokenss, 1))
            target_masks = (y_padded != nlc_data.PAD_ID).astype(np.int32)

            batch += (x_padded, source_masks, y_padded, target_masks)

            rewards, actions_dist, actions = [], [], []
            source_tokenss, target_tokenss = [], []

            yield batch

    # for residuals
    x_padded = np.array(padded(source_tokenss, FLAGS.num_layers)).T
    source_masks = (x_padded != nlc_data.PAD_ID).astype(np.int32)
    y_padded = np.array(padded(target_tokenss, 1))
    target_masks = (y_padded != nlc_data.PAD_ID).astype(np.int32)

    yield (np.array(rewards), np.concatenate(actions_dist, axis=1), np.array(actions),
           x_padded, source_masks, y_padded, target_masks)

    return

def setup_loss_critic(critic):
    # we are starting with critic.outputs symbol (after logistic layer)
    with tf.variable_scope("critic_update", initializer=tf.uniform_unit_scaling_initializer(1.0)):
        # loss setup
        # None to timestep
        critic.target_qt = tf.placeholder(tf.float32, shape=[None, None, critic.vocab_size],
                                            name="q_action_score")
        # p_actions is the target_token, and it's already [T, batch_size]
        # q_t needs to be expanded...

        # critic.outputs [T, batch_size, vocab_size]
        # let's populate (expand) target tokens to fill up qt (just like what we did with one-hot labels)

        critic.q_loss = tf.reduce_mean(tf.square(critic.outputs - critic.target_qt))  # Note: not adding lambda*C yet (variance)

        opt = nlc_model.get_optimizer(FLAGS.optimizer)(critic.learning_rate)

        # update
        params = tf.trainable_variables()
        gradients = tf.gradients(critic.q_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        #      self.gradient_norm = tf.global_norm(clipped_gradients)
        critic.gradient_norm = tf.global_norm(gradients)
        critic.param_norm = tf.global_norm(params)
        critic.updates = opt.apply_gradients(
            zip(clipped_gradients, params), global_step=critic.global_step)


def update_critic(sess, critic, q_values, p_actions, source_tokens, source_mask, target_mask=None):
    # similar to model.train() method!
    # note that we take out the target_masks (only used in setup_loss)
    # p_actions is the "target_tokens" (it's not action_dist)
    input_feed = {}
    input_feed[critic.source_tokens] = source_tokens
    input_feed[critic.source_mask] = source_mask
    input_feed[critic.target_tokens] = p_actions
    input_feed[critic.target_mask] = target_mask if target_mask is not None else np.ones_like(p_actions)

    one_hot_qt = tf.one_hot(p_actions, depth=critic.vocab_size).eval(feed_dict={},session=sess)
    target_qt = np.expand_dims(q_values, axis=2) * one_hot_qt
    input_feed[critic.target_qt] = target_qt  # [T, batch_size, vocab_size]
    # so now target_qt becomes one-hot encoding, but not with 1, but the target q at each position
    # after expanding dimension, it can broadcast multiply
    # TODO: make sure this part is correct though...after reduce_mean, it's still kinda big...
    # TODO: maybe it's the "rewards" problem (reward is too big)
    # TODO: maybe gradient clipping is NOT working!

    # grad_norm: 612.034
    # cost: 22950.4
    # param_norm: 72.1753

    input_feed[critic.keep_prob] = critic.keep_prob_config
    critic.set_default_decoder_state_input(input_feed, p_actions.shape[1])

    output_feed = [critic.updates, critic.gradient_norm, critic.q_loss, critic.param_norm]

    outputs = sess.run(output_feed, input_feed)

    return outputs[1], outputs[2], outputs[3]

def setup_actor_update(actor):
    # actor.critic_output = tf.placeholder(tf.float32, shape=[None, None, actor.vocab_size], name='critic_output')
    actor.action_gradients = tf.placeholder(tf.float32, [None, None, actor.vocab_size], name='action_gradients')
    # action_gradients is passed in by Q_network...
    opt = nlc_model.get_optimizer(FLAGS.optimizer)(actor.learning_rate)

    # update
    params = tf.trainable_variables()

    # http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html (DDPG update)
    gradients = tf.gradients(actor.losses, params, -actor.action_gradients)  # step 7: update
    # Not sure if I understood this part lol

    clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)

    # clip, then multiply, otherwise we are not learning the signals from critic
    # clipped_gradients: [T, batch_size, vocab_size]

    # updated_gradients = clipped_gradients * actor.critic_output
    # pass in as input

    actor.rl_gradient_norm = tf.global_norm(clipped_gradients)
    actor.rl_param_norm = tf.global_norm(params)

    actor.rl_updates = opt.apply_gradients(
        zip(clipped_gradients, params), global_step=actor.global_step)

def update_actor(sess, actor, q_values, p_actions, source_tokens, source_mask, target_tokens, target_mask):
    input_feed = {}
    input_feed[actor.source_tokens] = source_tokens
    input_feed[actor.source_mask] = source_mask
    input_feed[actor.target_tokens] = target_tokens
    input_feed[actor.target_mask] = target_mask



def train_critic(sess, actor, critic, delayed_actor, target_critic,
                 x_dev, y_dev, x_train, y_train, pretrain=False):
    # since actor is fixed, we can generate our own y_dev, y_train
    # use it to train such actor
    # for now...we encode at each turn (might be inefficient)

    i = 0

    # it's generating different source_tokens if we update our actor
    # NOTE: we are using delayed_actor p' to generate a sequence of actions
    for rewards, actions_dist, actions, source_tokens, \
        source_mask, target_tokens, target_mask in process_samples(sess, delayed_actor, x_train, y_train):

        # print(i)
        # print("%r %r %r" % (rewards.shape, source_tokens.shape, source_mask.shape))

        # action_dist = [T, batch_size, vocab_size]
        # rewards = [batch_size, T]
        # actions = [batch_size, T]  # remember target_tokens shape is [T, batch_size]
        # source_tokens = [T, batch_size]
        # target_tokens = [batch_size, T]

        # remember target_tokens here is NOT transposed

        # step 5
        critic_encoder_output = target_critic.encode(sess, target_tokens.T, target_mask.T)  # condition on ground truth
        critic_scores, _, _ = target_critic.decode(sess, critic_encoder_output, actions.T)  # actions needs to be transposed
        # print(critic_scores.shape) - assume it's [T, batch_size, vocab_size]

        q_values = rewards.T + np.sum(actions_dist * critic_scores, axis=2)
        # q_values shape: (32, 4)

        grad_norm, cost, param_norm = update_critic(sess, critic, q_values, actions.T, source_tokens, source_mask)
        print(grad_norm)
        print(cost)
        print(param_norm)

        # TODO: apply the graidnet (what's the shape of the gradient for actor model!?)
        # TODO: it needs to be aligned with critic's output... (batch_size, time_step, vocab_size)

        if not pretrain:
            # if not pretrain, we stop using a fixed actor...we update the actor as well
            pass
            # TODO: actor update (gradient might be wrong..or messed up)
            # update delayed_actor and target_critic

        # TODO: another check is...the tf.assign(), does it "decouple" and only assign the value?
        # TODO: or does it just do reference assign...

        sys.exit(0)




def set_params_values(source_params, target_params, sess, s_name, t_name, percentage=1.0, verbose=False):
    # assign source param values to target param values
    # tested, and works well!
    assignments = []

    for s_p, t_p in itertools.izip(source_params, target_params):
        assert s_p.name.replace(s_name, "") == t_p.name.replace(t_name, "")
        assignments.append(
            t_p.assign(s_p * percentage)
        )
    sess.run(assignments)


def train():
    global vocab, rev_vocab
    print("Preparing data in %s" % FLAGS.data_dir)
    path_2_ptb_data = FLAGS.data_dir + "/ptb_data"

    x_train = "{}/train.ids.x".format(path_2_ptb_data)
    y_train = "{}/train.ids.y".format(path_2_ptb_data)

    x_dev = "{}/valid.ids.x".format(path_2_ptb_data)
    y_dev = "{}/valid.ids.y".format(path_2_ptb_data)

    vocab_path = "{}/vocab.dat".format(path_2_ptb_data)

    # source_tokens and target_tokens are transposed
    source_tokens, source_mask, target_tokens, target_mask = build_data(fnamex="{}/train.ids.x".format(path_2_ptb_data),
                                                                        fnamey="{}/train.ids.y".format(path_2_ptb_data),
                                                                        num_layers=FLAGS.num_layers,
                                                                        max_seq_len=FLAGS.max_seq_len)
    vocab, rev_vocab = nlc_data.initialize_vocabulary(vocab_path)

    vocab_size = len(vocab)
    print("Vocabulary size: %d" % vocab_size)

    with tf.Session() as sess:
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        with tf.variable_scope("actor") as actor_vs:
            model = create_model(vocab_size, False, actor_vs.name)
            setup_actor_update(model)
        with tf.variable_scope("critic") as critic_vs:
            critic = create_model(vocab_size, False, critic_vs.name)
            setup_loss_critic(critic)
        with tf.variable_scope("delayed_actor") as delayed_actor_vs:
            delayed_actor = create_model(vocab_size, False, delayed_actor_vs.name)
            setup_actor_update(delayed_actor)
        with tf.variable_scope("target_critic") as target_critic_vs:
            target_critic = create_model(vocab_size, False, target_critic_vs.name)
            setup_loss_critic(target_critic)

        # if there is not model to restore, we initialize all of them
        # otherwise, we only need to restore ONCE for everything.
        if not restore_models(sess, model):
            initialize_models(sess, model)  # this should initialize all variables..
            # initialize_models(sess, delayed_actor)
            # initialize_models(sess, critic)
            # initialize_models(sess, target_critic)

        # by doing this, we are assigning embeddings as well
        # thinking about how critic's embeddings can make sense
        actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_vs.name)
        delayed_actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=delayed_actor_vs.name)
        critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=critic_vs.name)
        target_critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_critic_vs.name)

        if not FLAGS.rl_only:
            model = train_seq2seq(model, sess, x_dev, y_dev, x_train, y_train)  # pre-train actor

        # assign model's parameter values to delayed_actor
        set_params_values(actor_variables, delayed_actor_variables, sess, "actor", "delayed_actor")

        # assign critic's initial parameter values to target_critic
        set_params_values(critic_variables, target_critic_variables, sess, "critic", "target_critic")

        if not FLAGS.sup_only:
            # TODO: 2. actor (policy) is just the first seq2seq, build critic (2nd seq2seq)
            # TODO:    write training procedure for pretraining critic

            # pre-train critic
            train_critic(sess, model, critic, delayed_actor, target_critic,
                         x_dev, y_dev, x_train, y_train)

            # TODO: 3. write Algorithm 1 in paper


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
