# Architecture

## Contents

## Chain Pipelines

Chain pipelines provide a way to run any number and combination of other pipelines, automatically splitting them
into the correct tile size and passing the output on to the next stage.

## Worker Pool

The worker pool is a process pool that manages one or more worker processes for each device (typically a GPU).
