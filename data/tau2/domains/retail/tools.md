# Available tools

---

## calculate

```
Calculate the result of a mathematical expression.

Args:
    expression: The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.

Returns:
    The result of the mathematical expression.

Raises:
    ValueError: If the expression is invalid.
```

---

## cancel_pending_order

```
Cancel a pending order.

Args:
    order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
    reason: The reason for cancellation, which should be either 'no longer needed' or 'ordered by mistake'.

Returns:
    Order: The order details after the cancellation.
```

---

## exchange_delivered_order_items

```
Exchange items in a delivered order to new items of the same product type.

Args:
    order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
    item_ids: The item ids to be exchanged, each such as '1008292230'. There could be duplicate items in the list.
    new_item_ids: The item ids to be exchanged for, each such as '1008292230'.
                 There could be duplicate items in the list. Each new item id should match the item id
                 in the same position and be of the same product.
    payment_method_id: The payment method id to pay or receive refund for the item price difference,
                     such as 'gift_card_0000000' or 'credit_card_0000000'.

Returns:
    Order: The order details after the exchange.

Raises:
    ValueError: If the order is not delivered.
    ValueError: If the items to be exchanged do not exist.
    ValueError: If the new items do not exist or do not match the old items.
    ValueError: If the number of items to be exchanged does not match.
```

---

## find_user_id_by_email

```
Find user id by email. If the user is not found, the function will return an error message.

Args:
    email: The email of the user, such as 'something@example.com'.

Returns:
    str: The user id if found, otherwise an error message.

Raises:
    ValueError: If the user is not found.
```

---

## find_user_id_by_name_zip

```
Find user id by first name, last name, and zip code. If the user is not found, the function
will return an error message.

Args:
    first_name: The first name of the customer, such as 'John'.
    last_name: The last name of the customer, such as 'Doe'.
    zip: The zip code of the customer, such as '12345'.

Returns:
    str: The user id if found, otherwise an error message.

Raises:
    ValueError: If the user is not found.
```

---

## get_item_details

```
Get the inventory details of an item.

Args:
    item_id: The item id, such as '6086499569'. Be careful the item id is different from the product id.

Returns:
    Variant: The item details.

Raises:
    ValueError: If the item is not found.
```

---

## get_order_details

```
Get the status and details of an order.

Args:
    order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.

Returns:
    Order: The order details.

Raises:
    ValueError: If the order is not found.
```

---

## get_product_details

```
Get the inventory details of a product.

Args:
    product_id: The product id, such as '6086499569'. Be careful the product id is different from the item id.

Returns:
    Product: The product details.

Raises:
    ValueError: If the product is not found.
```

---

## get_user_details

```
Get the details of a user, including their orders.

Args:
    user_id: The user id, such as 'sara_doe_496'.

Returns:
    User: The user details.

Raises:
    ValueError: If the user is not found.
```

---

## list_all_product_types

```
List the name and product id of all product types.
Each product type has a variety of different items with unique item ids and options.
There are only 50 product types in the store.

Returns:
    str: A JSON string mapping product names to their product IDs, sorted alphabetically by name.
```

---

## modify_pending_order_address

```
Modify the shipping address of a pending order.

Args:
    order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
    address1: The first line of the address, such as '123 Main St'.
    address2: The second line of the address, such as 'Apt 1' or ''.
    city: The city, such as 'San Francisco'.
    state: The state, such as 'CA'.
    country: The country, such as 'USA'.
    zip: The zip code, such as '12345'.

Returns:
    Order: The order details after the modification.

Raises:
    ValueError: If the order is not pending.
```

---

## modify_pending_order_items

```
Modify items in a pending order to new items of the same product type

Args:
    order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
    item_ids: The item ids to be modified, each such as '1008292230'. There could be duplicate items in the list.
    new_item_ids: The item ids to be modified for, each such as '1008292230'. There could be duplicate items in the list. Each new item id should match the item id in the same position and be of the same product.
    payment_method_id: The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'.

Returns:
    Order: The order details after the modification.

Raises:
    ValueError: If the order is not pending.
    ValueError: If the items to be modified do not exist.
    ValueError: If the new items do not exist or do not match the old items.
    ValueError: If the number of items to be modified does not match.
```

---

## modify_pending_order_payment

```
Modify the payment method of a pending order.

Args:
    order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
    payment_method_id: The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'.

Returns:
    Order: The order details after the modification.

Raises:
    ValueError: If the order is not pending.
    ValueError: If the payment method does not exist.
    ValueError: If the payment history has more than one payment.
    ValueError: If the new payment method is the same as the current one.
```

---

## modify_user_address

```
Modify the default address of a user.

Args:
    user_id: The user id, such as 'sara_doe_496'.
    address1: The first line of the address, such as '123 Main St'.
    address2: The second line of the address, such as 'Apt 1' or ''.
    city: The city, such as 'San Francisco'.
    state: The state, such as 'CA'.
    country: The country, such as 'USA'.
    zip: The zip code, such as '12345'.

Returns:
    User: The user details after the modification.

Raises:
    ValueError: If the user is not found.
```

---

## return_delivered_order_items

```
Return some items of a delivered order.
The order status will be changed to 'return requested'.

Args:
    order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
    item_ids: The item ids to be returned, each such as '1008292230'. There could be duplicate items in the list.
    payment_method_id: The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'.

Returns:
    Order: The order details after requesting the return.

Raises:
    ValueError: If the order is not delivered.
    ValueError: If the payment method is not the original payment method or a gift card.
    ValueError: If the items to be returned do not exist.
```

---

## transfer_to_human_agents

```
Transfer the user to a human agent, with a summary of the user's issue.

Args:
    summary: A summary of the user's issue.

Returns:
    A message indicating the user has been transferred to a human agent.
```
