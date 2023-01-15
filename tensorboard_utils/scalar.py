from tbparse import SummaryReader


def get_last_scalar(scalars, exp_id, exp_type, model, scalar_tag):
    df = scalars.where(scalars['tag'] == scalar_tag).dropna()
    grouped_df = df.loc[df.groupby(['dir_name'])["step"].idxmax()]
    id_mask = grouped_df['dir_name'].str.contains(exp_id)
    type_mask = grouped_df['dir_name'].str.contains(exp_type)
    model_mask = grouped_df['dir_name'].str.contains(model)
    return grouped_df[id_mask & type_mask & model_mask]['value'].values[0]


def get_first_scalar(scalars, exp_id, exp_type, model, scalar_tag):
    df = scalars.where(scalars['tag'] == scalar_tag).dropna()
    grouped_df = df.loc[df.groupby(['dir_name'])["step"].idxmin()]
    id_mask = grouped_df['dir_name'].str.contains(exp_id)
    type_mask = grouped_df['dir_name'].str.contains(exp_type)
    model_mask = grouped_df['dir_name'].str.contains(model)
    return grouped_df[id_mask & type_mask & model_mask]['value'].values[0]
