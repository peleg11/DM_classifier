import click
from Project import *


'''
Shmuel Atias 300987443
Dmitry Korkin 336377429
Shay Peleg 302725643
'''

@click.command()
@click.option("--train", "-t", help="Train file")
@click.option("--test", "-t1", help="Test file")
@click.option("--struct", "-str", help="Structure  file")
@click.option("--model", "-m", help="NaiveBase/SklearnNaiveBase/ID3/SklearnID3/KNN/K-Means")
@click.option("--discr", "-d",
              help="Type of discritization:equalwidth_pandas/equalfreaqncy_pandas/equalwidth/equalfreaqncy"
                   "/entropybinning_external")
@click.option("--bin", "-b", help="Num of bins", type=int)
@click.option("--st_nr", "-sn", help="None/standartization/normalization")
@click.option("--result", "-r", help="Result y/n")
@click.option("--matrix", "-mat", help="Matrix y/n")
@click.option("--report", "-rep", help="Report y/n")
def main(train, test, struct, model, discr, bin, st_nr, result, matrix, report):
    traindf = pd.read_csv(train)
    # click.echo(traindf.head())
    testdf = pd.read_csv(test)
    # click.echo(traindf.head())
    structdf = pd.read_csv(struct, sep=' ', header=None, names=['att', 'att_name', 'types'], usecols=[1, 2])

    # ////////////////////////////////
    if model == "NaiveBase":
        proj = NaiveBayesClassifier(structdf, traindf, testdf)
    if model == "SklearnNaiveBase":
        proj = SklearnNB(structdf, traindf, testdf)
    if model == "ID3":
        proj = ID3(structdf, traindf, testdf)
    if model == "SklearnID3":
        proj = SklearnID3(structdf, traindf, testdf)
    if model == "KNN":
        proj = KNNClassifier(structdf, traindf, testdf)
    if model == "K-Means":
        proj = KMeansClustering(structdf, traindf, testdf)
    proj.clean()
    # /////////////////////
    if st_nr == "None":
        pass
    if st_nr == "standartization":
        proj.standardization()
    if st_nr == "normalization":
        proj.normalization()

    proj.discretization(bin, discr)
    # ////////////////

    # /////////////////////////////
    proj.buildModel()
    # ///////////////////
    if result == "y":
        click.echo(proj.result())
        click.echo("\n")
    if matrix == "y":
        click.echo("Matrix:\n")
        click.echo(proj.matrix())
        click.echo("\n")
    if report == "y":
        click.echo("Report:\n")
        click.echo(proj.report())

    click.echo("\n Bye Bye Bye")

    # click.echo("\n Bye Bye Bye")


if __name__ == '__main__':
    main()
