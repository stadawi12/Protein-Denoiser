import click
import numpy as np

from analysis import class_occurence, train_unet, predict_unet
from multi_proteins import load_data, preproces_data, query_data, plot_data, gen_labs, mpi_gen_labs


classtype_opts = ['backbone', 'residue', 'custom']
preproces_opts = ['crop', 'scale', 'tile']
utility_opts = ['makelabels', 'querymaps', 'makeplots', 'train', 'predict']  # TODO more to come
arch_opts = ['large', 'medium', 'small']
loss_opts = ['scce', 'weighted_scce', 'focal', 'custom_weighted_scce', 'prob_scce', 'weighted_prob_scce']
callback_opts = ['checkpoints', 'metrics', 'confusion', 'maps', 'stopping', 'tests', 'trains', 'timestamp']

d_help = 'Path to a directory containing .mrc files for maps and corresponding labels. Each directory is to be ' \
         'named with an associated resolution, labels to be named with a suffix \'label\', i.e. 2.5 and 2.5label'
r_help = 'Resolutions to use data from (use multiple times for multiple resolutions)'
n_help = 'Number of samples to load in'
c_help = 'Type of classes: {}'.format(classtype_opts)
p_help = 'Type of preprocessing: {}'.format(preproces_opts)
s_help = 'If -p/--preprocess set to \'crop\', use this to specify shape of the cropped image'
th_help = 'If using tiles, this option sets the threshold of background for the map.'
bg_help = 'If using tiles with threshold, this option sets percentage of background for accepted tiles.'
nr_help = 'If true, normalises data between 0 and 1.'
q_help = 'Query class distribution in each individual map'
u_help = 'Choice of functionality to implement: {}'.format(utility_opts)
e_help = 'Number of epochs'
b_help = 'Batch size'
a_help = 'Type of architecture: {}'.format(arch_opts)
l_help = 'Loss type: {}'.format(loss_opts)
v_help = 'Evaluate predictions.'
h_help = 'Name of the trained model for prediciton.'
m_help = 'When using -u/--utility makelabels, -m/--mpi executes this in parallel mode.'
g_help = 'When using -u/--utility makelables, -g/--genmaps will generate synthetic maps as well as labels.'
ts_help = 'When using -u/--utility makelables, -ts/--twosigma will generate data based on two sigmas. See atm_to_map.'
cb_help = 'Choice of callbacks: {}'.format(callback_opts)
jl_help = 'Json resolution indicator when generating labels.'


# setup args
@click.command(name="ML Protein Toolbox")
@click.option('--datapath', '-d', help=d_help, type=str, default=None, required=True)
@click.option('--res', '-r', help=r_help, type=float, multiple=True, default=None)
@click.option('--n', '-n', help=n_help, type=int, default=None)
@click.option('--classes', '-c', help=c_help, default=classtype_opts[2],
              type=click.Choice(classtype_opts))
# preprocess args
@click.option('--preproces', '-p', help=p_help, default=preproces_opts[0],
              type=click.Choice(preproces_opts))
@click.option('--cshape', '-s', help=s_help, type=int, default=64)
@click.option('--threshold', '-th', help=th_help, type=float, default=None)
@click.option('--background', '-bg', help=bg_help, type=float, default=0.5)
@click.option('--normalise', '-nr', help=nr_help, type=bool, default=True, is_flag=True)
@click.option('--querymaps', '-q', help=q_help, type=bool, default=False, is_flag=True)
# train args
@click.option('--utility', '-u', help=u_help, default=utility_opts[1],
              type=click.Choice(utility_opts))
@click.option('--epochs', '-e', help=e_help, type=int, default=5)
@click.option('--batch', '-b', help=b_help, type=int, default=1)
@click.option('--arch', '-a', help=a_help, default=arch_opts[0],
              type=click.Choice(arch_opts))
@click.option('--loss', '-l', help=l_help, default=loss_opts[0],
              type=click.Choice(loss_opts))
@click.option('--callback', '-cb', help=cb_help, type=str, default=callback_opts[:-2], multiple=True)
# predict args
@click.option('--evaluate', '-v', help=v_help, type=bool, default=False, is_flag=True)
@click.option('--model', '-h', help=h_help, type=str, default="model_general_64x64x64_final.h5")
# genlabs opts
@click.option('--mpilabs', '-m', help=m_help, type=bool, default=False, is_flag=True)
@click.option('--genmaps', '-g', help=g_help, type=bool, default=False, is_flag=True)
@click.option('--two_sigma', '-ts', help=ts_help, type=bool, default=False, is_flag=True)
@click.option('--json_labs', '-jl', help=jl_help, type=bool, default=False, is_flag=True)
def run(datapath, res, n, classes, preproces, cshape, threshold, background, normalise, querymaps,
        utility, epochs, batch, arch, loss, callback,
        evaluate, model, mpilabs, genmaps, two_sigma, json_labs):

    # 0) CHECK IF IT'S GEN_LAB FIRST
    if utility == 'makelabels':

        custom_labs = True if classes == 'custom' else False
        if mpilabs:
            mpi_gen_labs(datapath, res=list(res), custom_labs=custom_labs, two_sigma=two_sigma, gen_maps=genmaps, jlabs=json_labs)
        else:
            gen_labs(datapath, res=list(res), custom_labs=custom_labs, two_sigma=two_sigma,
                     gen_maps=genmaps, jlabs=json_labs)
        exit(0)

    res = list(res)
    if len(res) == 0:
        raise RuntimeError("Please specify parameter -r/--res with the requested resolutions to be processed.")
    print('Resolutions:{}'.format(res))

    # 1) LOAD DATA
    if datapath:
        pred = utility == 'predict' and not evaluate    # does not load labels if predicting but not evaluating
        wght = loss == loss_opts[3]
        data = load_data(datapath, res, n, predict=pred, weights=wght)

        if not pred:
            # 2) SET UP LABELS TODO there are better ways to do this
            if classes == 'residue':
                labels = ['0', 'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN',
                          'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 21]
            elif classes == 'backbone':
                labels = ['0', '1', '2', '3']
            else:
                labels = []
                for d in data:
                   labels.extend(np.unique(d.lab))
                labels = np.unique(labels)
                #labels = np.unique([d.lab for d in data])

        # 3) PREPROCESS DATA
        data = preproces_data(data, mode=preproces, cshape=cshape, threshold=threshold, background=background,
                              norm=normalise)
    else:
        raise RuntimeError('Please provide path to your data with --data/-d parameter.')

    # 4) EXECUTE UTILITY
    if utility == 'querymaps':
        query_data(data, res=res)

    elif utility == 'makeplots':
        plot_data(data)

    elif utility == 'classweights':
        class_occurence(data)

    elif utility == 'train':

        train_unet(data, labels,  # train_mode=train_mode,
                   epochs=epochs, batch=batch, arch_mode=arch, loss_mode=loss, callbacks=list(callback),
                   querymaps=querymaps)  # use kwargs to change batch / loss etc.

    elif utility == 'predict':

        predict_unet(data, evaluate=evaluate, model_name=model)


if __name__ == '__main__':
    run()
