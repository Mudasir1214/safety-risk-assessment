import argparse

def str2bool(v):
    """Convert string to boolean (for argparse)."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Safety Risk Assessment - Multi-Model Video Analysis"
    )

    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="Path to the input video file."
    )
    parser.add_argument(
        "--show_results",
        type=str2bool,
        default=True,
        help="Whether to display live video results."
    )
    parser.add_argument(
        "--save_results",
        type=str2bool,
        default=True,
        help="Whether to save output video with annotations."
    )

    return parser.parse_args()
