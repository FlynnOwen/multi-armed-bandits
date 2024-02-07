import typer

app = typer.Typer()


@app.command()
def simulate(distirbution: str,
             num_simulations: int,
             num_bandits: int,
             parameter_distribution: str,
             serialize_file_path: str,
             print_results: bool,
             print_plots: bool) -> None:
    """
    Placeholder for the main function.

    Current arguments are ideas only.
    """
    pass  #noqa: PIE790


if __name__ == "__main__":
    pass
