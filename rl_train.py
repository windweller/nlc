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


def create_model(session, vocab_size, forward_only, model_name):
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

        print(token_highest_prob)
        print("turn %d" % i)

    return output_sent


# TODO: maybe test this, but should work fine
def decompose_reward(a, b):
    # a is ground truth, both are tokenized with padding
    # return shape: (time,)
    reward_hist = np.zeros((FLAGS.max_seq_len), dtype=np.float32)
    reward_gain = np.zeros((FLAGS.max_seq_len), dtype=np.float32)
    print(a)
    for i in range(1, FLAGS.max_seq_len):
        reward_hist[i] = compute_cer(a[:i], b[:i])
    print(b)
    print(reward_hist)
    reward_gain[1:] = np.diff(reward_hist)  # first reward is always 0
    print(reward_gain)
    return reward_gain


# TODO: can we turn this into a generator?
def process_samples(sess, actor, x, y):
    # this batch things together based on the batch size
    # in the end, we can just izip the arrays, and iterate on them
    rewards, actions_dist, actions = [], [], []
    source_tokenss, source_masks, target_tokenss, target_masks = [], [], [], []
    # actions: (time, batch_size, vocab) # condition on ground truth targets

    batches = []

    for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x, y, 1,
                                                                            FLAGS.num_layers):

        source_tokenss.append(source_tokens[0])
        source_masks.append(source_mask[0])
        target_tokenss.append(target_tokens)
        target_masks.append(target_mask[0])

        encoder_output = actor.encode(sess, source_tokens, source_mask)
        best_tok, tok_prob = decode_beam(actor, sess, encoder_output, 1)
        best_str = detokenize(best_tok, rev_vocab)[0]  # greedy

        if best_str == "":
            best_str = " "

        best_str_tokens, best_str_mask = tokenize(best_str, vocab)
        rewards.append(decompose_reward(target_tokens, best_str_tokens))

        actions.append(best_str_tokens)
        actions_dist.append(tok_prob)

        if len(source_tokenss) % FLAGS.batch_size == 0:
            batches.append((np.array(rewards), np.array(actions_dist), np.array(actions),
                            np.array(source_tokenss), np.array(source_masks), np.array(target_tokenss),
                            np.array(target_masks)))
            rewards, actions_dist, actions = [], [], []
            source_tokenss, source_masks, target_tokenss, target_masks = [], [], [], []

        break

    # for residuals
    batches.append((np.array(rewards), np.array(actions_dist), np.array(actions),
                    np.array(source_tokenss), np.array(source_masks), np.array(target_tokenss),
                    np.array(target_masks)))
    return batches


# TODO: one of the central problem/question is that BLEU is a post-op measure
# TODO: in order to improve, we must do beam-search on the fly
def pretrain_critic(sess, actor, critic, delayed_actor, target_critic,
                    x_dev, y_dev, x_train, y_train):
    # since actor is fixed, we can generate our own y_dev, y_train
    # use it to train such actor
    # for now...we encode at each turn (might be inefficient)

    # we need to save rewards

    # TODO: 5. continue with Algorithm 1
    # possible to batch since we don't need beam-decode

    batches = process_samples(sess, actor, x_train, y_train)

    # q = rewards + np.sum(self.delayed_policy.f_output(observations) *
    #                      self.target_critic.compute_reward_sa(observations, actions), axis=2)




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
            model = create_model(sess, vocab_size, False, actor_vs.name)
        with tf.variable_scope("critic") as critic_vs:
            critic = create_model(sess, vocab_size, False, critic_vs.name)
        with tf.variable_scope("delayed_actor") as delayed_actor_vs:
            delayed_actor = create_model(sess, vocab_size, False, delayed_actor_vs.name)
        with tf.variable_scope("target_critic") as target_critic_vs:
            target_critic = create_model(sess, vocab_size, False, target_critic_vs.name)

        initialize_models(sess, model)

        # by doing this, we are assigning embeddings as well
        # thinking about how critic's embeddings can make sense
        actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_vs.name)
        # for v in actor_variables:
        #     print(v.name)
        delayed_actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=delayed_actor_vs.name)
        critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=critic_vs.name)
        target_critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_critic_vs.name)

        if not FLAGS.rl_only:
            model = train_seq2seq(model, sess, x_dev, y_dev, x_train, y_train)  # pre-train actor

        # assign model's parameter values to delayed_actor
        set_params_values(actor_variables, delayed_actor_variables, sess, "actor", "delayed_actor")

        if not FLAGS.sup_only:
            # TODO: 2. actor (policy) is just the first seq2seq, build critic (2nd seq2seq)
            # TODO:    write training procedure for pretraining critic

            # pre-train critic
            pretrain_critic(sess, model, critic, delayed_actor, target_critic,
                            x_dev, y_dev, x_train, y_train)

            # TODO: 3. write Algorithm 1 in paper


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
