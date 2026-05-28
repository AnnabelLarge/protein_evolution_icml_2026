import os


def write_final_eval_results(args,
                             summary_stats: dict,
                             filename: str):
    to_write_prefix = {'RUN': args.training_wkdir}
    to_write = {**to_write_prefix, **summary_stats}
    
    with open(f'{args.logfile_dir}/{filename}','w') as g:
        for k, v in to_write.items():
            g.write(f'{k}\t{v}\n')
            



# def format_tag(top_layer_name, 
#                 layer_name):
#     # reformat the tags to be: tag1 | tag2 | tag3 | ... | tag_n/lowest_tag
#     prefix = top_layer_name.replace('/', ' | ')
#     suffix = layer_name
#     tag = f'{prefix}/{suffix}'
#     return tag
