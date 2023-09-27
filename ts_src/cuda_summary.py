import torch 

def cuda_summary(N=10):
  # Create a list to store tensor information
  tensor_info = []

  # Iterate through all tensors on the GPU
  for obj in dir(torch.cuda):
      if 'Tensor' in obj:
          tensor = getattr(torch.cuda, obj)
          tensor_info.append((obj, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()))

  # Sort the tensors by memory usage
  tensor_info.sort(key=lambda x: x[2], reverse=True)

  # Print the top N tensors (adjust N as needed)
  print(f"Top {N} tensors by memory usage:")
  total_allocated = 0
  for i, (tensor_name, allocated, max_allocated) in enumerate(tensor_info[:N], 1):
      total_allocated += allocated
      print(f"{i}. {tensor_name}: Allocated={allocated / 1024**2:.2f} MB, Max Allocated={max_allocated / 1024**2:.2f} MB")

  print(total_allocated)
