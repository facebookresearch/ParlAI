ents = []
with open('/private/home/ahm/ParlAI/data/MovieDialog/movie_dialog_dataset/entities.txt') as ent_read:
    for line in ent_read:
        line = line.strip()
        if len(line) > 0:
            ents.append(line)

ents.sort(key=lambda x: -len(x))
entd = {e: '_^{}^_'.format(i) for i, e in enumerate(ents)}
enti = {i: e for e, i in entd.items()}

root = '/private/home/ahm/ParlAI/data/MovieDialog/movie_dialog_dataset/task3_qarecs/'
for read_fn, write_fn in [
    ('task3_qarecs_train.txt', 'task3_qarecs_pipe_train.txt'),
    ('task3_qarecs_dev.txt', 'task3_qarecs_pipe_dev.txt'),
    ('task3_qarecs_test.txt', 'task3_qarecs_pipe_test.txt'),
]:
    with open(root + read_fn) as read, open(root + write_fn, 'w') as write:
        for line in read:
            if '\t' in line:
                split = line.split('\t')
                ys = ' ' + split[1].strip()
                if ',' in ys:
                    found = []
                    for e in ents:
                        se = ' ' + e
                        if se in ys:
                            ys = ys.replace(se, entd[e])
                            found.append(entd[e])
                    ys = ys.replace(',', '|')
                    if len(found) != ys.count('|') + 1:
                        raise RuntimeError('Unexpected: {}'.format(ys))
                    for f in found:
                        ys = ys.replace(f, enti[f])

                    split[1] = ys
                    line = '\t'.join(split) + '\n'

            write.write(line)
