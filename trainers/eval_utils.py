import data.image_utils as iut
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import hydra
import cv2
from PIL import Image

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('agg')

class project_utils():
    def __init__(self, all_scenarios=True) -> None:
        self.all_scenarios = all_scenarios

    def set_all_scenarios(self, all_scenarios):
        self.all_scenarios = all_scenarios

    def get_original_beta_weights(self):
        ''' Weights of an optimized MNL with all parameters'''
        return get_original_beta_weights(self.all_scenarios)

PU = project_utils() # Global variable containing reference information not in cfg


def set_project_utils_options(cfg):  
    # Weight references: 
    PU.set_all_scenarios(not cfg.data.no_barrier_scenario)
    
    #init_plot_options: 
    if cfg.trainer.plot.activate_TkAgg:
        matplotlib.use('TkAgg')


LOG = logging.getLogger(__name__)


def reference_models():
    '''  Write a list of fixed models, move this into unit_tests script - unused'''
    return {}


def get_original_beta_weights(all_scenarios=True):
    ''' Weights of an optimized MNL with all parameters'''
    if all_scenarios:
        return {
            'Intervention': 0.178,
            'Barrier': -0.208,
            'CrossingSignal': 0.615,
            'Man': 0.586,
            'Woman': 0.689,
            'Pregnant': 0.831,
            'Stroller': 0.891,
            'OldMan': 0.288,
            'OldWoman': 0.342,
            'Boy': 0.878,
            'Girl': 0.980,
            'Homeless': 0.382,
            'LargeWoman': 0.550,
            'LargeMan': 0.447,
            'Criminal': 0.116,
            'MaleExecutive': 0.562,
            'FemaleExecutive': 0.650,
            'FemaleAthlete': 0.730,
            'MaleAthlete': 0.613,
            'FemaleDoctor': 0.753,
            'MaleDoctor': 0.682,
            'Dog': 0.179,
            'Cat': 0.112
            }
    else: # scneario 1 only betas
        return {
            'Intervention': 0.341,
            'Barrier': 0.112, # technically 0 - removed when masked
            'CrossingSignal': 0.623,
            'Man': 0.785,
            'Woman': 0.932,
            'Pregnant': 0.978,
            'Stroller': 1.028,
            'OldMan': 0.396,
            'OldWoman': 0.501,
            'Boy': 1.146,
            'Girl': 1.294,
            'Homeless': 0.531,
            'LargeWoman': 0.735,
            'LargeMan': 0.557,
            'Criminal': 0.187,
            'MaleExecutive': 0.781,
            'FemaleExecutive': 0.922,
            'FemaleAthlete': 1.007,
            'MaleAthlete': 0.855,
            'FemaleDoctor': 0.949,
            'MaleDoctor': 0.845,
            'Dog': 0.291,
            'Cat': 0.222
            }


def generate_scene_unit_tests():
    #'Write up manual scenes here. 2 values to create full binary scenario'
    scenes = []
    # 'On the Edge, Case where details Man vs ManXX count'
    scenes.append({
             'Saved': (0, 1),
             'Intervention': (0, 1),
             'Barrier': (0, 1),
             'CrossingSignal': (1, 0),
             'Man': (1, 0), 'Woman': (0, 0), 'Pregnant': (0, 0), 'Stroller': (0, 0), 'OldMan': (0, 0),
             'OldWoman': (1, 0), 'Boy': (1, 0), 'Girl': (0, 0), 'Homeless': (0, 0), 'LargeWoman': (0, 0), 'LargeMan': (0, 0),
             'Criminal': (0, 0), 'MaleExecutive': (0, 1), 'FemaleExecutive': (0, 0), 'FemaleAthlete': (0, 0),
             'MaleAthlete': (0, 1), 'FemaleDoctor': (0, 0), 'MaleDoctor': (0, 3), 'Dog': (0, 0), 'Cat': (0, 0)
             })
    return scenes

def get_ratio_tuples():
    # Ratios of interest to compare between models - currently all ratios based on a single reference (more effective)
    meaningul_ratios = [('Woman', 'Girl'), ('Man', 'Boy'), ('LargeWoman', 'FemaleAthlete'), ('LargeMan', 'MaleAthlete'), ('Boy', 'Girl'), ('OldMan', 'OldWoman'), ('Homeless', 'Criminal'), ('Dog', 'Cat')] 
    return meaningul_ratios


def get_weight_statistics(model, layer_name, X_cols, Masked_cols, normalizing_feature, device, original_weights=None):
    'Return General values concerning the beta parameters of a model and an original betas vector'
    if original_weights is None:
        original_weights = PU.get_original_beta_weights()
        original_weights = torch.tensor(
            [original_weights[X_col] for X_col in X_cols]).to(device)
    normalizing_indice = int(
        np.where(np.array(X_cols) == normalizing_feature)[0][0])

    for name, param in model.named_parameters():
        if layer_name in name:
            weights = param[0]
            break
        
    if Masked_cols:
        'Turn off the weights of the masked columns and Intervention (= ASC)'
        weights = weights.detach()
        for i,X_col in enumerate(X_cols):
            if X_col in Masked_cols:
                weights[i] = torch.zeros(weights[i].shape).to(weights[i].device)
                original_weights[i] = torch.zeros(weights[i].shape).to(weights[i].device)

    abs_difference = torch.abs(original_weights-weights)
    rel_error = abs_difference/original_weights
    normalized_original_weights = original_weights / \
        original_weights[normalizing_indice]
    normalized_weight = weights/weights[normalizing_indice]
    normalized_abs_difference = torch.abs(
        normalized_original_weights-normalized_weight)
    normalized_rel_error = abs_difference/normalized_original_weights

    weight_stats = {}
    ''' Add all beta metrics individually '''
    for i, col in enumerate(X_cols):
        weight_stats.update({
            f'beta_abs/{col}': abs_difference[i],
            f'beta_rel/{col}': rel_error[i],
            f'ratio_abs/{col}': normalized_abs_difference[i],
            f'ratio_rel/{col}': normalized_rel_error[i]
        })

    ''' Add Summary Metrics '''
    weight_stats.update({
            'beta_abs/max': torch.max(abs_difference),
            'beta_abs/total': torch.sum(abs_difference),
            'ratio_abs/max': torch.max(normalized_abs_difference),
            'ratio_abs/total': torch.sum(normalized_abs_difference),
            'beta_rel/max': torch.max(rel_error),
            'beta_rel/total': torch.sum(rel_error),
            'ratio_rel/max': torch.max(normalized_rel_error),
            'ratio_rel/total': torch.sum(normalized_rel_error)})

    return weight_stats


def get_unit_scene_tensors(X_cols, device):
    scenes = generate_scene_unit_tests()
    tensor_list = []
    for scene in scenes:
        tuples_list = [scene[X_col] for X_col in X_cols]
        tensor_scene = list(map(list, tuples_list))
        tensor_list.append(torch.tensor(tensor_scene).t())
    return torch.stack(tensor_list, dim=0).to(device)  # shape [N, 2, N_vars]


def get_beta_contributions(X_cols, input, device, trainer=None):
    'Return utility values from each beta parameters * input'
    outputs_dict = {}
    original_weights = PU.get_original_beta_weights()
    original_weights = torch.tensor(
        [original_weights[X_col] for X_col in X_cols]).to(device)
    X, *_ = input
    X = X.to(device).squeeze()

    def return_outputs(X, weights, name):
        outputs = X*weights
        return {name+'/'+X_col: outputs[..., i].detach()
                for i, X_col in enumerate(X_cols)}

    original_dict = return_outputs(X, original_weights, 'Original')
    outputs_dict.update(original_dict)
    if trainer is not None:
        X = trainer.masker.mask_tabular(X)
        masked_dict = return_outputs(X, original_weights, 'Masked')
        outputs_dict.update(masked_dict)
        for name, param in trainer.model.named_parameters():
            if 'utility' in name:
                model_weights = param[0]
                model_dict = return_outputs(X, model_weights, 'Model')
                outputs_dict.update(model_dict)
                break
    return outputs_dict


def get_r_beta_metrics(representations, beta_outputs_dict, input, X_cols, masked_list, device):
    ''' Gets multiple metrics comparing beta*input values vs R (image NN representation term) '''
    X, *_ = input
    X = X.to(device).squeeze()
    input_dict = {X_col: X[..., i] for i, X_col in enumerate(X_cols)}
    ''' Get average R contributions '''
    r_beta_dict = {'R_L': representations[:, 0],
                   'R_R': representations[:, 1],
                   'R_diff': torch.abs(representations[:, 0]-representations[:, 1])}
    if masked_list is not None:
        ''' Get R's  where a masked input value appears, and normalize it on number of masked values '''
        I_var_selection = [i for i, X_col in enumerate(
            X_cols) if X_col in masked_list]
        masked_inputs = X[..., I_var_selection]  # (batch, 2 , n_masked)
        masked_per_entry = masked_inputs.sum(dim=-1)  # (batch,2)
        I_r_selection = np.where(
            masked_per_entry.sum(dim=-1).cpu() != 0)[0]  # (selected)
        selected_r_terms = representations[I_r_selection]  # (selected, 2)
        # easiest is to sum both sides contributions :
        total_masked_entries = masked_per_entry[I_r_selection].sum(dim=-1)
        normalized_r_terms = selected_r_terms.sum(
            dim=-1)/total_masked_entries  # (selected)

        r_beta_dict.update({'R_normalized': normalized_r_terms,
                            'R_selected': selected_r_terms.sum(dim=-1),
                            })
        ''' Get diff or R's selected with original '''
        masked_output_orig = torch.stack(
            [beta_outputs_dict['Original/'+X_col] for X_col in X_cols if X_col in masked_list], dim=-1)  # (batch, 2, n_masked)

        # (selected, 2)
        selected_output_orig = masked_output_orig[I_r_selection].sum(dim=-1)
        diff_output_r = torch.abs(torch.abs(selected_r_terms[:, 0]-selected_r_terms[:, 1]) - torch.abs(
            selected_output_orig[:, 0]-selected_output_orig[:, 1]))  # (selected)
        r_beta_dict.update({'Origin_diff': diff_output_r})
        r_beta_dict.update(
            {'ASC_Orig': beta_outputs_dict['Original/Intervention'][:, 1]})
        if 'Model/Intervention' in beta_outputs_dict.keys():
            r_beta_dict.update(
                {'ASC_Model': beta_outputs_dict['Model/Intervention'][:, 1]})
            r_beta_dict.update(
                {'ASC_diff': beta_outputs_dict['Original/Intervention'][:, 1]-beta_outputs_dict['Model/Intervention'][:, 1]})
    return r_beta_dict


def log_r_beta_metrics(step_dicts, iteration, cloud_logger=None):
    ''' Creates and logs boxplots from step_dicts containing R_beta metrics '''
    beta_metrics = [step_dict['r_beta'] for step_dict in step_dicts]
    metric_names = beta_metrics[0].keys()
    data_plot = []
    for metric_name in metric_names:
        beta_metric = torch.cat([beta_metric[metric_name].unsqueeze(0)
                                 for beta_metric in beta_metrics], dim=1)
        data_plot.append(beta_metric.view(-1).cpu().numpy())
    plt.boxplot(np.array(data_plot)) # Occasionally bugs - TODO check if needed
    plt.xticks(np.arange(len(data_plot))+1, metric_names)
    if cloud_logger=='clearml':
        from clearml import Task
        task = Task.current_task()
        logger = task.get_logger()
        task.logger.report_matplotlib_figure(
            title="Beta_R_stats",
            series="R_boxplots",
            iteration=0,
            figure=plt,
        )
    elif cloud_logger=='wandb': # not working for debug mode: TODO check on Clearml, or saved figure
        import wandb
        # wandb.log({'Beta_R_stats':plt})
        # wandb.log({'Beta_R_stats':plt.figure()})
    plt.savefig('beta_r_plot.png')


def get_heatmaps_manually(trainer, input, loss_type='l2'):
    ''' Input contains X, images and labels, => prepare labels for desired Target Class'''
    # define the hook function
    hooked_layer = trainer.model.return_saliency_layer()
    hooked_outputs = {}
    def getOutputs_hook(name):
        # the hook signature
        def hook(model, input, output):
            hooked_outputs[name] = output
        return hook

    # register the hook to the desired layer(s)
    h = hooked_layer.register_forward_hook(getOutputs_hook('convolution'))

    # evaluate the trainer.model and get the output
    outputs, labels, size = trainer.infer_model(input)
    outputs, labels = trainer.model.preprocess_shapes(outputs, labels)
    losses = trainer.criterion(outputs, labels, trainer.model)
    # loss = losses[loss_type]
    loss = torch.nn.MSELoss(reduction=None)(outputs['detected'], labels['X'])
    print(loss)
    # compute the saliency map using the hook output
    grads = torch.autograd.grad(loss[1], hooked_outputs['convolution'])[0]
    pooled_grads = torch.mean(grads, dim=(2,3), keepdim=True)
    hooked_outputs['convolution'] = hooked_outputs['convolution'].cpu()
    pooled_grads = pooled_grads.cpu()
    # import pdb; pdb.set_trace()
    saliency_map = hooked_outputs['convolution'] * pooled_grads
    saliency_map = torch.sum(saliency_map, dim=1, keepdim=True)
    return saliency_map

def plot_heatmaps(saliency_maps, images=None, index=0):
        # # plot the saliency map
    if images is None:
        saliency_map = nn.functional.relu(saliency_map)
        saliency_map /= torch.max(saliency_map)
        salient_image = saliency_map[0][0].detach().numpy()
        plt.imshow(salient_image, cmap=plt.cm.hot)
        plt.axis('off')
        plt.show()
    else:
        try:
            image = images[index]
            image = image.squeeze(0).cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            # image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            # image = np.clip(image, 0, 255).astype(np.uint8)

            # Overlay saliency map on top of the original image
            saliency_maps = nn.functional.relu(saliency_maps)
            saliency_maps /= torch.max(saliency_maps)
            print(saliency_maps.shape)
            saliency_map = saliency_maps[index][0].detach().numpy()

            # saliency_map = np.array(saliency_map)[:,:,np.newaxis]
            print(saliency_map.shape)
            cm = plt.get_cmap('jet')
            saliency_map = cm(saliency_map)
            saliency_map = Image.fromarray((saliency_map * 255).astype(np.uint8))
            saliency_map = saliency_map.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
            # overlayed_image = np.concatenate((image, saliency_map), axis=2)

            overlayed_image = 0.2*np.array(saliency_map)[:,:,:3] + image*0.8
            # overlayed_image = Image.blend(Image.fromarray(image.astype(np.uint8)), saliency_map, alpha=0.4)
            # Plot the overlayed image
            fig, ax = plt.subplots(figsize=(8, 8))
            # ax.imshow(saliency_map)
            ax.imshow(overlayed_image)
            ax.axis('off')
            plt.show()
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()


def return_ratios(model, variables, stds, column_tuples, layer_name):
    # Retun the ratios and stds of pairwise variables listed in column_tuples at layer_name of the model
    # Print dataframe and saves into cloud_logger wandb ou clearml
    xs = model.named_parameters()
    names = []
    params = []
    for name, param in xs:
        if layer_name in name:
            params.append(param)
            names.append(name)

    ratios = []
    stds = []
    ratios_names = []
    for col1, col2 in column_tuples:
        ratios.append(params[0].data[0, variables.index(col1)] /
                      params[0].data[0, variables.index(col2)])
        ratios_names.append('Ratio ' + col1 + '/' + col2)
        stds.append(np.sqrt((params[1].data[0, variables.index(col1)] / params[0].data[0, variables.index(col1)])**2 +
                            (params[1].data[0, variables.index(col2)] / params[0].data[0, variables.index(col2)])**2))
    return ratios, stds, ratios_names

def log_Ratios(model, variables, stds, layer_name, cloud_logger=None, Title='Beta Ratios'):
    column_tuples = get_ratio_tuples()
    ratio_variables, ratio_stds, ratio_names = return_ratios(model, variables, stds, column_tuples, layer_name)
    log_Table(ratio_variables, ratio_stds, ratio_names, layer_name, cloud_logger, Title)


def log_Table(variables, stds, variables_names, layer_name, cloud_logger=None, Title='Beta Parameters'):
    # Print dataframe and saves into cloud_logger wandb ou clearml
    LOG.info('─' * 15 + 'Layer {}'.format(layer_name) + '-'*15)
    LOG.info('Variable\t\tValue\tstd'.expandtabs(16))
    LOG.info('─' * 55)
    for var, value, std in zip(variables_names, variables, stds):
        LOG.info(f'{var:<16}\t\t{value:.3f}\t\t{std:.3f}')
    df = pd.DataFrame(
            { 'Variable': [var for var in variables_names],
            'Value': [value for value in variables],
            'std': [std for std in stds]})
    if cloud_logger == 'clearml':
        from clearml import Task
        task = Task.current_task()
        logger = task.get_logger()
        logger.report_table(title= Title,
                            series='End epoch', table_plot=df)
    elif cloud_logger=='wandb':
        # log df to wandb
        import wandb
        wandb.log({Title : wandb.Table(dataframe=df)})


def return_Betas(model, variables, stds, layer_name):
    xs = model.named_parameters()
    names = []
    params = []
    for name, param in xs:
        if layer_name in name:
            params.append(param)
            names.append(name)
    return params[0].data[0], stds, variables


def log_Betas(model, variables, stds, layer_name, cloud_logger=None):
    # Print dataframe and saves into cloud_logger wandb ou clearml
    betas, stds, variables_names = return_Betas(model, variables, stds, layer_name)
    log_Table(betas.cpu(), stds, variables_names, layer_name, cloud_logger)



def get_utility_Hessian(cfg, trainer, layer_name='utility.weight'):
    """Get Hessian of Linear layer.

    Args:
        trainer: trainer with a train_dataloader and model
        layer_name: name of the utility layer for getting std values of parameters

    Returns:
        type: Hessian Diagonal

    """
    method = cfg.trainer.eval.hessian_method
    if method not in ['diagonal', 'naive', 'greedy']:
        raise NotImplementedError

    batch_hessians = []
    if cfg.trainer.cuda:
        torch.cuda.empty_cache()
    for batch_idx, input in enumerate(trainer.test_dataloader):
        # trainer.model.to('cpu')
        # trainer.train_device = 'cpu'
        step_dict = trainer.hessian_step(input)
        loss = step_dict['loss']
        batch_size = step_dict['size']
        xs = trainer.model.named_parameters()  # Get parameters we want gradient from
        names = []
        params = []
        for name, param in xs:
            if layer_name == name:  # (Only utility layer)
                params.append(param)
                names.append(name)
        xs = params
        ys = loss
        # print(xs)
        first_order_grads = torch.autograd.grad(
            ys, xs, retain_graph=True, create_graph=True, allow_unused=True)  # first order gradient
        # Prepare for next gradient pass
        first_order_grads = torch.cat([x.view(-1) for x in first_order_grads])
        # print(first_order_grads.shape)

        second_order_grads = []
        for grads in first_order_grads:
            s_grads = torch.autograd.grad(grads, xs, retain_graph=True)
            # print(grads.shape)
            # print(s_grads[0].shape)
            # concatenate on single dimension all given parameters
            s_grads = torch.cat([x.view(-1) for x in s_grads])
            second_order_grads.append(s_grads)
        # Stack rows of Hessian (param x param)
        second_order_grads = torch.stack(second_order_grads, axis=0)
        # print(second_order_grads.shape)
        # grad reduced by batch_size, recover full gradient value.
        hessian = second_order_grads * batch_size
        batch_hessians.append(hessian)
        # print('batch_idx and size {} / {}'.format(batch_idx, batch[0].shape[0]))

    # Sum all batch contributions to Hessian
    Hessian = torch.sum(torch.stack(batch_hessians, axis=0), axis=0)
    # inv_Hess = torch.inverse(Hessian)
    Hessian = Hessian.cpu().numpy()
    ' Stop here if not greedy method: '
    if method in ['diagonal', 'naive']:
        if method == 'diagonal':
            Hessian = np.diag(np.diag(Hessian))
        inv_Hess = np.linalg.inv(Hessian)
        stds = [inv_Hess[i][i]**0.5 for i in range(inv_Hess.shape[0])]
        return stds, []
    # exclude = [i for i in range(Hessian.shape[0]) if math.isclose(Hessian[i][i], 0, abs_tol=1)]

    'Start with non-singular 2x2, then greedy create non-singular hessian'

    exclude = np.arange(Hessian.shape[0])
    found_pair = False
    for i in exclude:
        next_i = False
        for j in range(i+1, len(exclude)):
            try:
                inv_Hess = torch.inverse(
                    torch.from_numpy(Hessian[[i, j], :][:, [i, j]]))
                break   # If good j with i found, get out of both loops
            except:
                pass    # If not, go to next j-
            next_i = True  # If not break found, go to next i

        if not next_i:
            found_pair = True
            break

    if not found_pair:
        # if failed stop function here
        return np.zeros(Hessian.shape[0]), np.arange(Hessian.shape[0])

    ''' Greedy search all indices (with torch instability, removed bad values at next iterations)'''
    np.delete(exclude, [i, j])
    for i in np.arange(Hessian.shape[0]):
        try:
            position = np.where(exclude == i)
            exclude = np.delete(exclude, position)
            Hessian_temp = np.delete(Hessian, exclude, axis=0)
            Hessian_temp = np.delete(Hessian_temp, exclude, axis=1)
            inv_Hess = torch.inverse(torch.from_numpy(Hessian_temp))
        except RuntimeError:
            LOG.warning('Add {} failed'.format(i))
            exclude = np.insert(exclude,  np.argmax(
                position), i)  # keep it in exclude list

    # print('Final exclude list: {}'.format(exclude))

    Hessian = np.delete(Hessian, exclude, axis=0)
    Hessian = np.delete(Hessian, exclude, axis=1)
    # Make use of torch inverse instability! (Singular with numpy or scipy)
    inv_Hess = torch.inverse(torch.from_numpy(Hessian))
    # Remove unstable Hessian values (very big))
    exclude_2nd = [i for i in range(Hessian.shape[0]) if inv_Hess[i][i] > 1000]
    # exclude_2nd = [] # Remove unstable Hessian values (very big))
    # print(exclude)
    # stds = [inv_Hess[i][i]**0.5 for i in range(inv_Hess.shape[0])] # Diagonal of inverse
    # print('STDs: {}'.format(np.array(stds).flatten()))
    Hessian = np.delete(Hessian, exclude_2nd, axis=0)
    Hessian = np.delete(Hessian, exclude_2nd, axis=1)
    # Check with other methods: Done
    # inv_Hess = torch.inverse(torch.from_numpy(Hessian))
    # stds = [inv_Hess[i][i]**0.5 for i in range(inv_Hess.shape[0])] # Diagonal of inverse
    # print('STDs: {}'.format(np.array(stds).flatten()))
    # inv_Hess = spla.solve(Hessian, np.eye(Hessian.shape[0]))
    # stds = [inv_Hess[i][i]**0.5 for i in range(inv_Hess.shape[0])] # Diagonal of inverse
    # print('STDs: {}'.format(np.array(stds).flatten()))
    inv_Hess = np.linalg.inv(Hessian)  # Do real numpy inverse
    # Diagonal of inverse
    stds = [inv_Hess[i][i]**0.5 for i in range(inv_Hess.shape[0])]
    #
    # print(len(exclude_2nd))
    # print(len(exclude))
    for i in np.sort(exclude_2nd):
        stds = np.insert(np.array(stds), i, 'NaN')
    for i in np.sort(exclude):
        stds = np.insert(np.array(stds), i, 'NaN')
    # print('STDs: {}'.format(np.array(stds).flatten()))
    return stds, exclude
