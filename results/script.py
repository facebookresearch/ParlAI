import matplotlib.pyplot as plt


def read_data_into_float_list(name):
    my_file = open(name, "r")
    data = my_file.read()
    data_into_list = [float(i) for i in data.split("\n")[:-1]]
    return data_into_list


def read_data_into_str_list(name):
    my_file = open(name, "r")
    data = my_file.read()
    data_into_list = data.split("\n")[:-1]
    return data_into_list


def read_tensor_into_float_list(name):
    my_file = open(name, "r")
    data = my_file.read()
    data_into_list = [
        float(i.split(",")[0].split('(')[1]) for i in data.split("\n")[:-1]
    ]
    return data_into_list


def plot_mem(file1, file2, save_name):
    parlai_mem = read_data_into_float_list(file1)
    triton_mem = read_data_into_float_list(file2)

    plt.plot(range(len(parlai_mem)), parlai_mem, color='green')
    plt.plot(range(len(triton_mem)), triton_mem, color='blue')
    plt.savefig(save_name)


def count_diff_sentences(file1, file2):
    diff_count = 0
    parlai_txt = read_data_into_str_list(file1)
    triton_txt = read_data_into_str_list(file2)
    for i in range(len(parlai_txt)):
        if parlai_txt[i] != triton_txt[i]:
            diff_count += 1
            print(parlai_txt[i])
            print(triton_txt[i])
            print("-----------------------")
    print(
        f"{diff_count} out of {len(parlai_txt)} sentences differ, %{diff_count/len(parlai_txt)}"
    )


plot_mem("parlai_mem.txt", "triton_mem.txt", 'mem_usage.png')
count_diff_sentences("parlai.txt", "triton.txt")
data = read_tensor_into_float_list("diff.txt")
print(f'average of sums of differences is {sum(data)/len(data)}')
