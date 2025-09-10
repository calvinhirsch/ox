pub struct Zip<I: Iterator, const N: usize>([I; N]);

impl<I: Iterator, const N: usize> Iterator for Zip<I, N> {
    type Item = [I::Item; N];

    fn next(&mut self) -> Option<Self::Item> {
        let next_items: [_; N] = self.0.each_mut().map(Iterator::next);

        if next_items.iter().any(|item| item.is_none()) {
            // If one iterator is depleted, make sure all others are too
            debug_assert!(
                next_items.iter().any(|item| item.is_none()),
                "LOD chunk iterators passed to LODSplitter have different numbers of chunks."
            );
            return None;
        }

        Some(next_items.map(|i| i.unwrap()))
    }
}

pub fn zip<I: Iterator, const N: usize>(iterators: [I; N]) -> Zip<I, N> {
    Zip(iterators)
}
