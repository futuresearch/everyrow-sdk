"""
Derive Example

Demonstrates using the derive operation to add a computed column.

Usage:
    python derive_example.py
"""

import asyncio

from pandas import DataFrame

from everyrow.ops import derive


async def main():
    data = DataFrame(
        [
            {"product": "Widget", "price": 10.00, "quantity": 5},
            {"product": "Gadget", "price": 25.50, "quantity": 3},
            {"product": "Gizmo", "price": 7.25, "quantity": 10},
        ]
    )

    print("Input data:")
    print(data.to_string())

    result = await derive(
        input=data,
        expressions={
            "total": "price * quantity",
        },
    )

    print("\nWith derived 'total' column:")
    print(result.data.to_string())


if __name__ == "__main__":
    asyncio.run(main())
