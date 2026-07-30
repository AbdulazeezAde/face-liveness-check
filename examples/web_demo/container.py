"""Container entry point for the local-only browser integration example."""

from server import main


if __name__ == "__main__":
    # Docker must listen on all container interfaces; docker-compose maps this
    # port to 127.0.0.1 on the host, keeping the browser demo local-only.
    main(host="0.0.0.0")
